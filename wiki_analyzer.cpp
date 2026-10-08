#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>


// ===================================================================
// CHARACTER TABLE
// ===================================================================
struct LowerTable {
    unsigned char t[256];
    constexpr LowerTable() : t{} {
        for (int c = 'a'; c <= 'z'; ++c) t[c] = static_cast<unsigned char>(c);
        for (int c = 'A'; c <= 'Z'; ++c) t[c] = static_cast<unsigned char>(c + 32);
    }
};
static constexpr LowerTable LOWER{};


// ===================================================================
// HASHING
// ===================================================================
static inline uint64_t load64(const char *p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

static inline uint64_t mix(uint64_t a, uint64_t b) {
    __uint128_t r = static_cast<__uint128_t>(a ^ 0xa0761d6478bd642full) *
                    (b ^ 0xe7037ed1a0b428dbull);
    return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
}

static inline uint64_t hashKey(const char *key, uint32_t len) {
    uint64_t h = len;
    for (uint32_t i = 0; i < len; i += 8)
        h = mix(h, load64(key + i));
    return mix(h, 0x8ebc6af09c88c6e3ull);
}

static inline bool keyEquals(const char *a, const char *b, uint32_t len) {
    for (uint32_t i = 0; i < len; i += 8)
        if (load64(a + i) != load64(b + i)) return false;
    return true;
}


// ===================================================================
// FLAT COUNTER
// ===================================================================
class Arena {
    static constexpr size_t BLOCK = 1 << 20;
    std::vector<std::unique_ptr<char[]>> blocks;
    char *cur = nullptr;
    size_t left = 0;

public:
    const char *store(const char *src, uint32_t len) {
        size_t need = (static_cast<size_t>(len) + 7) & ~size_t(7);
        if (need > left) {
            size_t sz = std::max(BLOCK, need);
            blocks.emplace_back(new char[sz]);
            cur = blocks.back().get();
            left = sz;
        }
        std::memcpy(cur, src, need);
        const char *r = cur;
        cur += need;
        left -= need;
        return r;
    }

    void clear() {
        blocks.clear();
        cur = nullptr;
        left = 0;
    }
};

class FlatCounter {
public:
    struct Entry {
        uint64_t hash;
        const char *key;
        uint32_t len;
        uint64_t count;
    };

private:
    std::vector<Entry> slots;
    size_t mask;
    size_t used = 0;
    Arena arena;

    void grow() {
        std::vector<Entry> old(slots.size() * 2, Entry{0, nullptr, 0, 0});
        old.swap(slots);
        mask = slots.size() - 1;
        for (const Entry &e : old) {
            if (!e.key) continue;
            size_t i = e.hash & mask;
            while (slots[i].key) i = (i + 1) & mask;
            slots[i] = e;
        }
    }

public:
    explicit FlatCounter(size_t capacity = 1 << 14)
        : slots(capacity, Entry{0, nullptr, 0, 0}), mask(capacity - 1) {}

    void add(uint64_t hash, const char *key, uint32_t len, uint64_t c) {
        if ((used + 1) * 2 > slots.size()) grow();
        size_t i = hash & mask;
        while (true) {
            Entry &e = slots[i];
            if (!e.key) {
                e = Entry{hash, arena.store(key, len), len, c};
                ++used;
                return;
            }
            if (e.hash == hash && e.len == len && keyEquals(e.key, key, len)) {
                e.count += c;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    size_t size() const { return used; }
    const std::vector<Entry> &entries() const { return slots; }

    void clear() {
        std::fill(slots.begin(), slots.end(), Entry{0, nullptr, 0, 0});
        used = 0;
        arena.clear();
    }
};


// ===================================================================
// STOP LIST
// ===================================================================
static const std::unordered_set<std::string_view> STOP_LIST = {
    // HTML/XML Entities & Tags
    "quot", "gt", "lt", "amp", "ref", "br", "span", "div", "small",
    "ul", "wul", "html", "class", "style", "align", "nbsp", "http",
    "https", "www", "px", "org", "com",

    // Wikipedia Templates & Fields
    "date", "user", "web", "url", "title", "name", "category", "cite",
    "archive", "talk", "access", "link", "news", "coibot", "domain",
    "utc", "otherlinks", "center", "page", "color", "website",
    "publisher", "base", "article", "background", "redirect",
    "language", "id", "basedomain", "baseip", "wikipedia",

    // Single-letter noise (often template parameters)
    // We keep "a" and "i" as they are real words.
    "s", "x", "u", "l", "d", "c", "b", "f", "e", "w", "m", "p", "de", "r", "t"
};


// ===================================================================
// TOKENIZER
// ===================================================================
class Tokenizer {
    std::vector<char> buf = std::vector<char>(256);

public:
    void countWords(std::string_view text, FlatCounter &freq) {
        auto *p = reinterpret_cast<const unsigned char *>(text.data());
        auto *end = p + text.size();
        while (true) {
            while (p < end && !LOWER.t[*p]) ++p;
            if (p == end) break;
            const unsigned char *start = p;
            while (p < end && LOWER.t[*p]) ++p;

            uint32_t len = static_cast<uint32_t>(p - start);
            if (len + 8 > buf.size()) buf.resize((len + 8) * 2);
            char *w = buf.data();
            for (uint32_t i = 0; i < len; ++i) w[i] = static_cast<char>(LOWER.t[start[i]]);
            std::memset(w + len, 0, 8);
            freq.add(hashKey(w, len), w, len, 1);
        }
    }
};


// ===================================================================
// SHARDED COUNTER
// ===================================================================
class ShardedWordCounter {
private:
    static const size_t NUM_SHARDS = 256;
    struct Shard {
        FlatCounter map;
        std::mutex mtx;
    };
    std::vector<Shard> shards;

    static size_t shardIndex(uint64_t hash) { return hash >> 56; }

    template <typename F>
    void forEachWord(F f) {
        for (auto &s : shards) {
            std::lock_guard<std::mutex> lock(s.mtx);
            for (const auto &e : s.map.entries()) {
                if (!e.key) continue;
                std::string_view w(e.key, e.len);
                if (STOP_LIST.count(w)) continue;
                f(w, e.count);
            }
        }
    }

public:
    ShardedWordCounter() : shards(NUM_SHARDS) {}

    void merge(const FlatCounter &local, size_t startShard) {
        const auto &entries = local.entries();

        std::vector<uint32_t> offsets(NUM_SHARDS + 1, 0);
        for (const auto &e : entries)
            if (e.key) ++offsets[shardIndex(e.hash) + 1];
        for (size_t i = 0; i < NUM_SHARDS; ++i) offsets[i + 1] += offsets[i];
        std::vector<const FlatCounter::Entry *> sorted(offsets[NUM_SHARDS]);
        std::vector<uint32_t> fill(offsets.begin(), offsets.end() - 1);
        for (const auto &e : entries)
            if (e.key) sorted[fill[shardIndex(e.hash)]++] = &e;

        for (size_t k = 0; k < NUM_SHARDS; ++k) {
            size_t i = (startShard + k) % NUM_SHARDS;
            if (offsets[i] == offsets[i + 1]) continue;
            auto &s = shards[i];
            std::lock_guard<std::mutex> lock(s.mtx);
            for (uint32_t j = offsets[i]; j < offsets[i + 1]; ++j) {
                const auto *e = sorted[j];
                s.map.add(e->hash, e->key, e->len, e->count);
            }
        }
    }

    // Memory-efficient getTopK using a min-heap
    std::vector<std::pair<std::string, uint64_t>> getTopK(size_t k) {
        using WordPair = std::pair<std::string_view, uint64_t>;
        struct Compare {
            bool operator()(const WordPair &a, const WordPair &b) {
                return a.second > b.second; // Min-heap
            }
        };
        std::priority_queue<WordPair, std::vector<WordPair>, Compare> minHeap;

        forEachWord([&](std::string_view w, uint64_t c) {
            if (minHeap.size() < k) {
                minHeap.push({w, c});
            } else if (k > 0 && c > minHeap.top().second) {
                minHeap.pop();
                minHeap.push({w, c});
            }
        });

        std::vector<std::pair<std::string, uint64_t>> topItems;
        topItems.reserve(minHeap.size());
        while (!minHeap.empty()) {
            topItems.emplace_back(std::string(minHeap.top().first), minHeap.top().second);
            minHeap.pop();
        }
        std::sort(topItems.begin(), topItems.end(), [](auto &a, auto &b) {
            return a.second > b.second;
        });
        return topItems;
    }

    size_t uniqueWords() {
        size_t total = 0;
        forEachWord([&](std::string_view, uint64_t) { ++total; });
        return total;
    }

    uint64_t totalWords() {
        uint64_t total = 0;
        forEachWord([&](std::string_view, uint64_t c) { total += c; });
        return total;
    }
};


// ===================================================================
// PARSING
// ===================================================================
class RangeReader {
    int fd;
    size_t fileSize;
    std::vector<char> buf;
    size_t bufBegin = 0, bufLen = 0;

    bool readAt(size_t offset, size_t len, char *dst) {
        while (len > 0) {
            ssize_t n = pread(fd, dst, len, static_cast<off_t>(offset));
            if (n <= 0) return false;
            dst += n;
            offset += static_cast<size_t>(n);
            len -= static_cast<size_t>(n);
        }
        return true;
    }

public:
    RangeReader(int fd, size_t fileSize) : fd(fd), fileSize(fileSize) {}

    std::string_view load(size_t begin, size_t len) {
        len = std::min(len, fileSize - begin);
        if (buf.size() < len) buf.resize(len);
        bufBegin = begin;
        bufLen = readAt(begin, len, buf.data()) ? len : 0;
        return {buf.data(), bufLen};
    }

    bool extend(size_t more, std::string_view &view) {
        size_t start = bufBegin + bufLen;
        more = std::min(more, fileSize - start);
        if (more == 0) return false;
        buf.resize(bufLen + more);
        if (!readAt(start, more, buf.data() + bufLen)) return false;
        bufLen += more;
        view = {buf.data(), bufLen};
        return true;
    }
};

constexpr size_t READ_SLACK = 1ul << 20;  // 1 MB

static size_t processRange(RangeReader &reader, size_t begin, size_t end,
                           Tokenizer &tok, FlatCounter &freq) {
    std::string_view buf = reader.load(begin, end - begin + READ_SLACK);
    const size_t rangeLen = end - begin;
    size_t articles = 0;
    size_t pos = 0;
    while (true) {
        size_t pageStart = buf.find("<page>", pos);
        if (pageStart == std::string_view::npos || pageStart >= rangeLen) break;
        size_t pageEnd = buf.find("</page>", pageStart);
        if (pageEnd == std::string_view::npos) {
            if (!reader.extend(std::max(READ_SLACK, buf.size()), buf)) break;
            continue;
        }

        std::string_view page = buf.substr(pageStart, pageEnd - pageStart);
        size_t textStart = page.find("<text");
        if (textStart != std::string_view::npos) {
            textStart = page.find('>', textStart);
            if (textStart != std::string_view::npos) {
                ++textStart;
                size_t textEnd = page.find("</text>", textStart);
                if (textEnd != std::string_view::npos) {
                    tok.countWords(page.substr(textStart, textEnd - textStart), freq);
                    ++articles;
                }
            }
        }
        pos = pageEnd + 7;
    }
    return articles;
}

struct SharedState {
    int fd = -1;
    size_t fileSize = 0;
    std::atomic<size_t> nextOffset{0};
    std::atomic<size_t> bytesProcessed{0};
    std::atomic<size_t> articlesProcessed{0};
    ShardedWordCounter counter;

    std::mutex doneMtx;
    std::condition_variable doneCv;
    unsigned workersDone = 0;
};

constexpr size_t RANGE_SIZE = 32ul * 1024 * 1024;
constexpr size_t LOCAL_FLUSH_AT = 1ul << 19;

void worker(SharedState &st, size_t workerId) {
    Tokenizer tok;
    FlatCounter local(1 << 16);
    RangeReader reader(st.fd, st.fileSize);
    const size_t size = st.fileSize;

    while (true) {
        size_t begin = st.nextOffset.fetch_add(RANGE_SIZE);
        if (begin >= size) break;
        size_t end = std::min(begin + RANGE_SIZE, size);

        st.articlesProcessed += processRange(reader, begin, end, tok, local);
        st.bytesProcessed += end - begin;

        if (local.size() >= LOCAL_FLUSH_AT) {
            st.counter.merge(local, workerId * 37);
            local.clear();
        }
    }
    st.counter.merge(local, workerId * 37);

    {
        std::lock_guard<std::mutex> lock(st.doneMtx);
        ++st.workersDone;
    }
    st.doneCv.notify_all();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <enwiki-latest-pages-articles.xml> [top_k=100]\n";
        return 1;
    }

    std::string filename = argv[1];
    size_t topK = (argc >= 3) ? std::stoul(argv[2]) : 100;

    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "Cannot open " << filename << "\n";
        return 1;
    }
    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        std::cerr << "Cannot stat " << filename << "\n";
        return 1;
    }
    size_t fileSize = static_cast<size_t>(sb.st_size);

    auto start = std::chrono::steady_clock::now();

    SharedState st;
    st.fd = fd;
    st.fileSize = fileSize;

    unsigned int nThreads = std::max(2u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < nThreads; ++i)
        workers.emplace_back(worker, std::ref(st), i);

    {
        std::unique_lock<std::mutex> lock(st.doneMtx);
        while (!st.doneCv.wait_for(lock, std::chrono::milliseconds(500),
                                   [&] { return st.workersDone == nThreads; })) {
            std::cout << "Read " << st.bytesProcessed / (1024 * 1024)
                      << " MB | Articles processed: " << st.articlesProcessed
                      << "\r" << std::flush;
        }
    }
    for (auto &t : workers)
        t.join();

    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>(end - start);

    std::cout << "\n\nProcessed " << fileSize / (1024.0 * 1024.0)
              << " MB in " << std::fixed << std::setprecision(2) << dur.count()
              << " seconds.\n" << std::defaultfloat;
    std::cout << "Total articles: " << st.articlesProcessed.load() << "\n";
    std::cout << "Total words: " << st.counter.totalWords() << "\n";
    std::cout << "Unique words: " << st.counter.uniqueWords() << "\n\n";

    auto top = st.counter.getTopK(topK);
    std::cout << "Top " << topK << " Words:\n";
    std::cout << std::string(40, '=') << "\n";
    int rank = 1;
    for (auto &[w, c] : top)
        std::cout << std::setw(4) << rank++ << ". " << std::setw(25) << w
                  << " : " << c << "\n";

    close(fd);
    return 0;
}
