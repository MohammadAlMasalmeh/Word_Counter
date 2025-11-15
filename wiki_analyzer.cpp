#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <queue>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <condition_variable>
#include <atomic>
#include <string_view>
#include <unordered_set> 


class WikiXMLParser {
public:
    struct Article {
        std::string title;
        std::string text;
    };

    static std::vector<Article> parseChunk(std::string_view xmlChunk) {
        std::vector<Article> articles;
        size_t pos = 0;
        while (true) {
            size_t pageStart = xmlChunk.find("<page>", pos);
            if (pageStart == std::string::npos) break;
            size_t pageEnd = xmlChunk.find("</page>", pageStart);
            if (pageEnd == std::string::npos) break;
            std::string_view pageContent = xmlChunk.substr(pageStart, pageEnd - pageStart);
            size_t titleStart = pageContent.find("<title>");
            size_t titleEnd = pageContent.find("</title>");
            std::string title;
            if (titleStart != std::string::npos && titleEnd != std::string::npos)
                title = std::string(pageContent.substr(titleStart + 7, titleEnd - titleStart - 7));
            size_t textStart = pageContent.find("<text");
            if (textStart != std::string::npos) {
                textStart = pageContent.find('>', textStart);
                if (textStart != std::string::npos) {
                    ++textStart;
                    size_t textEnd = pageContent.find("</text>", textStart);
                    if (textEnd != std::string::npos) {
                        std::string text = std::string(pageContent.substr(textStart, textEnd - textStart));
                        articles.push_back({std::move(title), std::move(text)});
                    }
                }
            }
            pos = pageEnd + 7;
        }
        return articles;
    }
};

class TextProcessor {
private:
    // Create a static set of words to ignore.
    static const std::unordered_set<std::string> STOP_LIST;

public:
    static void tokenizeAndCount(const std::string &text,
                                 std::unordered_map<std::string, uint64_t> &freq) {
        std::string word;
        word.reserve(32);

        for (char c : text) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                word += static_cast<char>(std::tolower(c));
            } else if (!word.empty()) {
                // ** MODIFICATION: Only check the stop list **
                if (STOP_LIST.find(word) == STOP_LIST.end()) {
                    ++freq[word];
                }
                word.clear();
            }
        }
        // Check the last word
        if (!word.empty() && STOP_LIST.find(word) == STOP_LIST.end()) {
            ++freq[word];
        }
    }
};

// ** Define the static STOP_LIST outside the class **
const std::unordered_set<std::string> TextProcessor::STOP_LIST = {
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
// SHARDED COUNTER (Memory-efficient getTopK)
// ===================================================================
class ShardedWordCounter {
private:
    static const size_t NUM_SHARDS = 256;
    struct Shard {
        std::unordered_map<std::string, uint64_t> map;
        std::mutex mtx;
    };
    std::vector<Shard> shards;

    size_t shardIndex(const std::string &w) const {
        return std::hash<std::string>{}(w) % NUM_SHARDS;
    }

public:
    ShardedWordCounter() : shards(NUM_SHARDS) {}

    void merge(const std::unordered_map<std::string, uint64_t> &local) {
        for (const auto &[w, c] : local) {
            auto &s = shards[shardIndex(w)];
            std::lock_guard<std::mutex> lock(s.mtx);
            s.map[w] += c;
        }
    }

    // Memory-efficient getTopK using a min-heap
    std::vector<std::pair<std::string, uint64_t>> getTopK(size_t k) {
        using WordPair = std::pair<std::string, uint64_t>;
        struct Compare {
            bool operator()(const WordPair &a, const WordPair &b) {
                return a.second > b.second; // Min-heap
            }
        };
        std::priority_queue<WordPair, std::vector<WordPair>, Compare> minHeap;

        for (auto &s : shards) {
            std::lock_guard<std::mutex> lock(s.mtx);
            for (const auto &[w, c] : s.map) {
                if (minHeap.size() < k) {
                    minHeap.push({w, c});
                } else if (c > minHeap.top().second) {
                    minHeap.pop();
                    minHeap.push({w, c});
                }
            }
        }

        std::vector<WordPair> topItems;
        topItems.reserve(minHeap.size());
        while (!minHeap.empty()) {
            topItems.push_back(minHeap.top());
            minHeap.pop();
        }
        std::sort(topItems.begin(), topItems.end(), [](auto &a, auto &b) {
            return a.second > b.second;
        });
        return topItems;
    }

    size_t uniqueWords() {
        size_t total = 0;
        for (auto &s : shards) {
            std::lock_guard<std::mutex> lock(s.mtx);
            total += s.map.size();
        }
        return total;
    }

    uint64_t totalWords() {
        uint64_t total = 0;
        for (auto &s : shards) {
            std::lock_guard<std::mutex> lock(s.mtx);
            for (auto &[w, c] : s.map)
                total += c;
        }
        return total;
    }
};

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv_pop;
    std::condition_variable cv_push;
    bool done = false;
    size_t maxSize;

public:
    ThreadSafeQueue(size_t max) : maxSize(max) {}

    void push(T item) {
        {
            std::unique_lock<std::mutex> lock(m);
            cv_push.wait(lock, [this] { return q.size() < maxSize || done; });
            if (done) return;
            q.push(std::move(item));
        }
        cv_pop.notify_one();
    }

    bool pop(T &out) {
        std::unique_lock<std::mutex> lock(m);
        cv_pop.wait(lock, [this] { return !q.empty() || done; });
        if (q.empty())
            return false;
        out = std::move(q.front());
        q.pop();
        lock.unlock();
        cv_push.notify_one();
        return true;
    }

    void setFinished() {
        {
            std::lock_guard<std::mutex> lock(m);
            done = true;
        }
        cv_pop.notify_all();
        cv_push.notify_all();
    }

    // size() method for monitoring
    size_t size() {
        std::lock_guard<std::mutex> lock(m);
        return q.size();
    }
};

void processBatch(const std::vector<WikiXMLParser::Article> &arts,
                  ShardedWordCounter &counter) {
    std::unordered_map<std::string, uint64_t> local;
    local.reserve(10000);
    for (auto &a : arts)
        TextProcessor::tokenizeAndCount(a.text, local);
    counter.merge(local);
}

void worker(ThreadSafeQueue<std::vector<WikiXMLParser::Article>> &queue,
            ShardedWordCounter &counter, std::atomic<size_t> &count) {
    std::vector<WikiXMLParser::Article> batch;
    while (queue.pop(batch)) {
        processBatch(batch, counter);
        count += batch.size();
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <enwiki-latest-pages-articles.xml> [top_k=100]\n";
        return 1;
    }

    std::string filename = argv[1];
    size_t topK = (argc >= 3) ? std::stoul(argv[2]) : 100;

    constexpr size_t CHUNK_SIZE = 250ul * 1024 * 1024; // 250 MB
    constexpr size_t BATCH_SIZE = 1000;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open " << filename << "\n";
        return 1;
    }

    auto start = std::chrono::steady_clock::now();

    unsigned int nThreads = std::max(2u, std::thread::hardware_concurrency());
    
    // Initialize queue with a max size (e.g., 2 batches per worker)
    ThreadSafeQueue<std::vector<WikiXMLParser::Article>> queue(nThreads * 2);
    
    ShardedWordCounter counter;
    std::atomic<size_t> articlesProcessed = 0;

    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < nThreads; ++i)
        workers.emplace_back(worker, std::ref(queue), std::ref(counter),
                             std::ref(articlesProcessed));

    std::string leftover;
    std::vector<WikiXMLParser::Article> batch;
    batch.reserve(BATCH_SIZE);

    size_t totalBytes = 0;

    std::vector<char> buffer(CHUNK_SIZE);
    while (file) {
        file.read(buffer.data(), CHUNK_SIZE);
        size_t bytesRead = file.gcount();
        if (!bytesRead)
            break;

        totalBytes += bytesRead;

        // More efficient string handling
        std::string current = std::move(leftover);
        current.append(std::string_view(buffer.data(), bytesRead));
        
        size_t pos = 0;
        while (true) {
            size_t pageStart = current.find("<page>", pos);
            if (pageStart == std::string::npos) {
                leftover.assign(current.data() + pos, current.size() - pos);
                break;
            }
            size_t pageEnd = current.find("</page>", pageStart);
            if (pageEnd == std::string::npos) {
                leftover.assign(current.data() + pageStart, current.size() - pageStart);
                break;
            }

            std::string_view pageView(current.data() + pageStart,
                                      pageEnd - pageStart + 7);
            auto arts = WikiXMLParser::parseChunk(pageView);
            for (auto &a : arts) {
                batch.push_back(std::move(a));
                if (batch.size() >= BATCH_SIZE) {
                    queue.push(std::move(batch));
                    batch.clear();
                    batch.reserve(BATCH_SIZE);
                }
            }
            pos = pageEnd + 7;
        }

        std::cout << "Read " << totalBytes / (1024 * 1024)
                  << " MB | Articles processed: " << articlesProcessed
                  << " | Queue size: " << queue.size() 
                  << "\r" << std::flush;
    }

    if (!batch.empty())
        queue.push(std::move(batch));

    file.close();
    queue.setFinished();

    for (auto &t : workers)
        t.join();

    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << "\n\nProcessed " << totalBytes / (1024.0 * 1024.0)
              << " MB in " << dur.count() << " seconds.\n";
    std::cout << "Total articles: " << articlesProcessed.load() << "\n";
    std::cout << "Total words: " << counter.totalWords() << "\n";
    std::cout << "Unique words: " << counter.uniqueWords() << "\n\n";

    auto top = counter.getTopK(topK);
    std::cout << "Top " << topK << " Words:\n";
    std::cout << std::string(40, '=') << "\n";
    int rank = 1;
    for (auto &[w, c] : top)
        std::cout << std::setw(4) << rank++ << ". " << std::setw(25) << w
                  << " : " << c << "\n";

    return 0;
}