#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif


// ===================================================================
// LOWERCASING
// ===================================================================
static constexpr size_t PAD = 64;

#ifdef TUNING
static size_t tunable(const char *name, size_t dflt) {
    const char *v = getenv(name);
    return v ? std::strtoull(v, nullptr, 10) : dflt;
}
#define TUNABLE(name, dflt) static const size_t name = tunable(#name, dflt)
#else
#define TUNABLE(name, dflt) static constexpr size_t name = dflt
#endif

static void lowerInto(const char *text, size_t n, uint8_t *out) {
    auto *src = reinterpret_cast<const uint8_t *>(text);
    size_t i = 0;
#if defined(__ARM_NEON)
    const uint8x16_t bit5 = vdupq_n_u8(0x20), a = vdupq_n_u8('a'), span = vdupq_n_u8(26);
    for (; i + 16 <= n; i += 16) {
        uint8x16_t x = vorrq_u8(vld1q_u8(src + i), bit5);
        uint8x16_t isLetter = vcltq_u8(vsubq_u8(x, a), span);
        vst1q_u8(out + i, vandq_u8(x, isLetter));
    }
#endif
    for (; i < n; ++i) {
        uint8_t x = src[i] | 0x20;
        out[i] = (static_cast<uint8_t>(x - 'a') < 26) ? x : 0;
    }
    std::memset(out + n, 0, PAD);
}

static inline uint64_t nonZeroMask64(const uint8_t *p) {
#if defined(__ARM_NEON)
    static const uint8_t bitsArr[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t bits = vld1q_u8(bitsArr);
    uint8x16_t v0 = vld1q_u8(p), v1 = vld1q_u8(p + 16), v2 = vld1q_u8(p + 32), v3 = vld1q_u8(p + 48);
    uint8x16_t t0 = vandq_u8(vtstq_u8(v0, v0), bits), t1 = vandq_u8(vtstq_u8(v1, v1), bits);
    uint8x16_t t2 = vandq_u8(vtstq_u8(v2, v2), bits), t3 = vandq_u8(vtstq_u8(v3, v3), bits);
    uint8x16_t s = vpaddq_u8(vpaddq_u8(t0, t1), vpaddq_u8(t2, t3));
    s = vpaddq_u8(s, s);
    return vgetq_lane_u64(vreinterpretq_u64_u8(s), 0);
#else
    uint64_t m = 0;
    for (int k = 0; k < 64; ++k)
        if (p[k]) m |= uint64_t(1) << k;
    return m;
#endif
}


// ===================================================================
// HASHING
// ===================================================================
static inline uint64_t load64(const uint8_t *p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

static inline uint64_t loadPartial(const uint8_t *p, uint32_t rem) {
    uint64_t v = load64(p);
    return rem >= 8 ? v : v & ((uint64_t(1) << (rem * 8)) - 1);
}

static inline uint64_t mix(uint64_t a, uint64_t b) {
    __uint128_t r = static_cast<__uint128_t>(a ^ 0xa0761d6478bd642full) *
                    (b ^ 0xe7037ed1a0b428dbull);
    return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
}

struct WordRef {
    uint64_t hash;
    uint64_t head;
    const uint8_t *tail;
    uint32_t len;
};

static inline WordRef makeWord(const uint8_t *p, uint32_t len) {
    WordRef w{0, loadPartial(p, len), p + 8, len};
    uint64_t h = mix(w.head, len);
    for (uint32_t i = 8; i < len; i += 8)
        h = mix(h, loadPartial(p + i, len - i));
    w.hash = h;
    return w;
}

static inline bool tailEquals(const uint8_t *a, const uint8_t *b, uint32_t tailLen) {
    for (uint32_t i = 0; i < tailLen; i += 8)
        if (loadPartial(a + i, tailLen - i) != loadPartial(b + i, tailLen - i)) return false;
    return true;
}


// ===================================================================
// FLAT COUNTER
// ===================================================================
class FlatCounter {
public:
    struct Entry {
        uint64_t hash;
        uint64_t head;
        uint64_t count;
        uint32_t tailOff;
        uint32_t len;
    };

private:
    std::vector<Entry> slots;
    size_t mask;
    size_t used = 0;
    std::vector<uint8_t> arena;

    void grow() {
        std::vector<Entry> old(slots.size() * 2, Entry{});
        old.swap(slots);
        mask = slots.size() - 1;
        for (const Entry &e : old) {
            if (!e.len) continue;
            size_t i = e.hash & mask;
            while (slots[i].len) i = (i + 1) & mask;
            slots[i] = e;
        }
    }

    uint32_t storeTail(const uint8_t *tail, uint32_t tailLen) {
        size_t off = arena.size();
        arena.resize(off + tailLen + 8);
        std::memcpy(arena.data() + off, tail, tailLen);
        return static_cast<uint32_t>(off);
    }

    Entry *find(const WordRef &w) {
        size_t i = w.hash & mask;
        while (true) {
            Entry &e = slots[i];
            if (!e.len) return &e;
            if (e.hash == w.hash && e.head == w.head && e.len == w.len &&
                (w.len <= 8 || tailEquals(arena.data() + e.tailOff, w.tail, w.len - 8)))
                return &e;
            i = (i + 1) & mask;
        }
    }

public:
    explicit FlatCounter(size_t capacity = 1 << 14)
        : slots(capacity, Entry{}), mask(capacity - 1) {}

    bool addIfRoom(const WordRef &w, size_t limit) {
        Entry *e = find(w);
        if (!e->len) {
            if (used >= limit) return false;
            uint32_t off = w.len > 8 ? storeTail(w.tail, w.len - 8) : 0;
            *e = Entry{w.hash, w.head, 0, off, w.len};
            ++used;
        }
        ++e->count;
        return true;
    }

    void addRecords(const uint64_t *data, size_t words) {
        while ((used + words / 3) * 2 > slots.size()) grow();
        for (size_t i = 0; i < words; i += 3 + recordTailWords(data[i + 2]))
            __builtin_prefetch(&slots[data[i] & mask]);
        for (size_t i = 0; i < words; i += 3 + recordTailWords(data[i + 2])) {
            uint32_t len = static_cast<uint32_t>(data[i + 2]);
            WordRef w{data[i], data[i + 1], reinterpret_cast<const uint8_t *>(data + i + 3), len};
            add(w, data[i + 2] >> 32);
        }
    }

    static size_t recordTailWords(uint64_t lenAndCount) {
        uint32_t len = static_cast<uint32_t>(lenAndCount);
        return len > 8 ? (len - 8 + 7) / 8 : 0;
    }

    void add(const WordRef &w, uint64_t c) {
        if ((used + 1) * 2 > slots.size()) grow();
        Entry *e = find(w);
        if (!e->len) {
            uint32_t off = w.len > 8 ? storeTail(w.tail, w.len - 8) : 0;
            *e = Entry{w.hash, w.head, 0, off, w.len};
            ++used;
        }
        e->count += c;
    }

    void remove(const WordRef &w) {
        Entry *e = find(w);
        if (e->len) e->count = 0;
    }

    WordRef ref(const Entry &e) const {
        return WordRef{e.hash, e.head, arena.data() + e.tailOff, e.len};
    }

    std::string word(const Entry &e) const {
        std::string s(e.len, '\0');
        std::memcpy(&s[0], &e.head, std::min<uint32_t>(e.len, 8));
        if (e.len > 8) std::memcpy(&s[8], arena.data() + e.tailOff, e.len - 8);
        return s;
    }

    size_t size() const { return used; }
    const std::vector<Entry> &entries() const { return slots; }

    void clear() {
        std::fill(slots.begin(), slots.end(), Entry{});
        used = 0;
        arena.clear();
    }
};


// ===================================================================
// STOP LIST
// ===================================================================
static const char *const STOP_LIST[] = {
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
    std::vector<uint8_t> low;

public:
    template <typename Sink>
    void countWords(std::string_view text, Sink &sink) {
        const size_t n = text.size();
        if (low.size() < n + PAD) low.resize(n + PAD + n / 2);
        lowerInto(text.data(), n, low.data());
        const uint8_t *p = low.data();

        auto emit = [&](size_t start, size_t end) {
            sink.add(makeWord(p + start, static_cast<uint32_t>(end - start)));
        };

        bool inWord = false;
        size_t wordStart = 0;
        uint64_t carry = 0;
        for (size_t base = 0; base < n; base += 64) {
            uint64_t m = nonZeroMask64(p + base);
            uint64_t prev = (m << 1) | carry;
            uint64_t starts = m & ~prev;
            uint64_t ends = ~m & prev;
            carry = m >> 63;

            while (true) {
                if (inWord) {
                    if (!ends) break;
                    emit(wordStart, base + __builtin_ctzll(ends));
                    ends &= ends - 1;
                    inWord = false;
                } else {
                    if (!starts) break;
                    wordStart = base + __builtin_ctzll(starts);
                    starts &= starts - 1;
                    inWord = true;
                }
            }
        }
        if (inWord) emit(wordStart, n);
    }
};


// ===================================================================
// PARTITIONED COUNTER
// ===================================================================
TUNABLE(PARTITION_BITS, 11); // 2048 partitions

class PartitionedCounter {
private:
    struct alignas(128) Shard {
        FlatCounter map{1 << 10};
        std::mutex mtx;
    };
    std::vector<Shard> shards;

public:
    static size_t shardIndex(uint64_t hash) { return hash >> (64 - PARTITION_BITS); }

public:
    struct Summary {
        uint64_t totalWords = 0;
        size_t uniqueWords = 0;
        std::vector<std::pair<std::string, uint64_t>> top;
    };

    PartitionedCounter() : shards((size_t(1) << PARTITION_BITS)) {}

    static size_t partitions() { return (size_t(1) << PARTITION_BITS); }

    void apply(size_t partition, const std::vector<uint64_t> &records) {
        auto &s = shards[partition];
        std::lock_guard<std::mutex> lock(s.mtx);
        s.map.addRecords(records.data(), records.size());
    }

    Summary summarize(size_t k) {
        for (const char *stop : STOP_LIST) {
            uint32_t len = static_cast<uint32_t>(std::strlen(stop));
            std::vector<uint8_t> buf(len + PAD, 0);
            std::memcpy(buf.data(), stop, len);
            WordRef w = makeWord(buf.data(), len);
            auto &s = shards[shardIndex(w.hash)];
            std::lock_guard<std::mutex> lock(s.mtx);
            s.map.remove(w);
        }

        using Ranked = std::pair<uint64_t, std::pair<size_t, const FlatCounter::Entry *>>;
        std::priority_queue<Ranked, std::vector<Ranked>, std::greater<Ranked>> minHeap;

        Summary sum;
        for (size_t si = 0; si < (size_t(1) << PARTITION_BITS); ++si) {
            auto &s = shards[si];
            std::lock_guard<std::mutex> lock(s.mtx);
            for (const auto &e : s.map.entries()) {
                if (!e.len || !e.count) continue;
                sum.totalWords += e.count;
                ++sum.uniqueWords;
                if (minHeap.size() < k) {
                    minHeap.push({e.count, {si, &e}});
                } else if (k > 0 && e.count > minHeap.top().first) {
                    minHeap.pop();
                    minHeap.push({e.count, {si, &e}});
                }
            }
        }

        while (!minHeap.empty()) {
            auto [count, where] = minHeap.top();
            sum.top.emplace_back(shards[where.first].map.word(*where.second), count);
            minHeap.pop();
        }
        std::sort(sum.top.begin(), sum.top.end(), [](auto &a, auto &b) {
            return a.second > b.second;
        });
        return sum;
    }
};


// ===================================================================
// LOCAL COUNTER
// ===================================================================
TUNABLE(HOT_BITS, 15);          // 32K slots (1 MB), filled to half
TUNABLE(OUTBOX_WORDS, 1024);    // 8 KB per partition buffer

class LocalCounter {
    FlatCounter hot{size_t(1) << HOT_BITS};
    PartitionedCounter &global;
    std::vector<std::vector<uint64_t>> outbox;

    void emit(const WordRef &w, uint64_t count) {
        size_t p = PartitionedCounter::shardIndex(w.hash);
        auto &ob = outbox[p];
        ob.push_back(w.hash);
        ob.push_back(w.head);
        ob.push_back(uint64_t(w.len) | (count << 32));
        for (uint32_t i = 8; i < w.len; i += 8)
            ob.push_back(loadPartial(w.tail + (i - 8), w.len - i));
        if (ob.size() >= OUTBOX_WORDS) {
            global.apply(p, ob);
            ob.clear();
        }
    }

public:
    explicit LocalCounter(PartitionedCounter &global)
        : global(global), outbox(PartitionedCounter::partitions()) {}

    void add(const WordRef &w) {
        if (!hot.addIfRoom(w, hot.entries().size() / 2)) emit(w, 1);
    }

    void flushHot() {
        for (const auto &e : hot.entries())
            if (e.len) emit(hot.ref(e), e.count);
        hot.clear();
    }

    void flushAll() {
        flushHot();
        for (size_t p = 0; p < outbox.size(); ++p) {
            if (outbox[p].empty()) continue;
            global.apply(p, outbox[p]);
            outbox[p].clear();
        }
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
                           Tokenizer &tok, LocalCounter &freq) {
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
    alignas(128) std::atomic<size_t> nextOffset{0};
    alignas(128) std::atomic<size_t> bytesProcessed{0};
    alignas(128) std::atomic<size_t> articlesProcessed{0};
    alignas(128) char pad_[128];
    PartitionedCounter counter;

    std::mutex doneMtx;
    std::condition_variable doneCv;
    unsigned workersDone = 0;
};

constexpr size_t RANGE_SIZE = 32ul * 1024 * 1024;

TUNABLE(HOT_RESET, 1); // flush the hot table after every range

void worker(SharedState &st) {
    Tokenizer tok;
    LocalCounter local(st.counter);
    RangeReader reader(st.fd, st.fileSize);
    const size_t size = st.fileSize;

    while (true) {
        size_t begin = st.nextOffset.fetch_add(RANGE_SIZE);
        if (begin >= size) break;
        size_t end = std::min(begin + RANGE_SIZE, size);

        st.articlesProcessed += processRange(reader, begin, end, tok, local);
        st.bytesProcessed += end - begin;
        if (HOT_RESET) local.flushHot();
    }
    local.flushAll();

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
        workers.emplace_back(worker, std::ref(st));

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

    auto summary = st.counter.summarize(topK);

    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>(end - start);

    std::cout << "\n\nProcessed " << fileSize / (1024.0 * 1024.0)
              << " MB in " << std::fixed << std::setprecision(2) << dur.count()
              << " seconds.\n" << std::defaultfloat;
    std::cout << "Total articles: " << st.articlesProcessed.load() << "\n";
    std::cout << "Total words: " << summary.totalWords << "\n";
    std::cout << "Unique words: " << summary.uniqueWords << "\n\n";

    std::cout << "Top " << topK << " Words:\n";
    std::cout << std::string(40, '=') << "\n";
    int rank = 1;
    for (auto &[w, c] : summary.top)
        std::cout << std::setw(4) << rank++ << ". " << std::setw(25) << w
                  << " : " << c << "\n";

    close(fd);
    return 0;
}
