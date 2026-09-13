#include "router.h"

#include "api_error.h"

#include <algorithm>
#include <cctype>

namespace audiocpp::api {

std::vector<std::string> split_path(const std::string & path) {
    std::vector<std::string> parts;
    std::string current;
    for (const char c : path) {
        if (c == '/') {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

bool match_route(const std::string & pattern, const std::string & path, PathParams & params) {
    const auto want = split_path(pattern);
    const auto got = split_path(path);
    if (want.size() != got.size()) {
        return false;
    }
    PathParams captured;
    for (size_t i = 0; i < want.size(); ++i) {
        const auto & segment = want[i];
        if (segment.size() >= 2 && segment.front() == '{' && segment.back() == '}') {
            const auto name = segment.substr(1, segment.size() - 2);
            if (got[i].empty()) {
                return false;
            }
            captured[name] = url_decode(got[i]);
            continue;
        }
        if (segment != got[i]) {
            return false;
        }
    }
    params = std::move(captured);
    return true;
}

std::string url_decode(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '+') {
            out.push_back(' ');
            continue;
        }
        if (value[i] == '%' && i + 2 < value.size() && std::isxdigit(static_cast<unsigned char>(value[i + 1])) &&
            std::isxdigit(static_cast<unsigned char>(value[i + 2]))) {
            const auto hex = value.substr(i + 1, 2);
            out.push_back(static_cast<char>(std::stoi(hex, nullptr, 16)));
            i += 2;
            continue;
        }
        out.push_back(value[i]);
    }
    return out;
}

std::unordered_map<std::string, std::string> parse_query(const std::string & query) {
    std::unordered_map<std::string, std::string> out;
    size_t start = 0;
    while (start < query.size()) {
        auto end = query.find('&', start);
        if (end == std::string::npos) {
            end = query.size();
        }
        const auto pair = query.substr(start, end - start);
        if (!pair.empty()) {
            const auto eq = pair.find('=');
            if (eq == std::string::npos) {
                out[url_decode(pair)] = "";
            } else {
                out[url_decode(pair.substr(0, eq))] = url_decode(pair.substr(eq + 1));
            }
        }
        start = end + 1;
    }
    return out;
}

std::string query_value(const std::string & query, const std::string & key) {
    const auto params = parse_query(query);
    const auto found = params.find(key);
    return found == params.end() ? std::string() : found->second;
}

void Router::add(std::string method, std::string pattern, RouteHandler handler) {
    routes_.push_back(Route{std::move(method), std::move(pattern), std::move(handler)});
}

bool Router::path_exists(const std::string & path) const {
    PathParams params;
    for (const auto & route : routes_) {
        if (match_route(route.pattern, path, params)) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> Router::methods_for(const std::string & path) const {
    std::vector<std::string> methods;
    PathParams params;
    for (const auto & route : routes_) {
        if (match_route(route.pattern, path, params) &&
            std::find(methods.begin(), methods.end(), route.method) == methods.end()) {
            methods.push_back(route.method);
        }
    }
    return methods;
}

minitts::server::HttpResponse Router::dispatch(const minitts::server::HttpRequest & request) const {
    PathParams params;
    for (const auto & route : routes_) {
        if (route.method == request.method && match_route(route.pattern, request.path, params)) {
            return route.handler(request, params);
        }
    }
    // The path exists but not for this verb: say so, rather than making a client
    // wonder whether they got the URL wrong.
    const auto methods = methods_for(request.path);
    if (!methods.empty()) {
        std::string allow;
        for (const auto & method : methods) {
            if (!allow.empty()) {
                allow += ", ";
            }
            allow += method;
        }
        auto response = error_response(405, request.method + " is not allowed on " + request.path +
                                                " (allowed: " + allow + ")",
                                       "invalid_request_error");
        response.headers["Allow"] = allow;
        return response;
    }
    return error_response(404, "unknown endpoint: " + request.path, "not_found");
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_ROUTER
// Direct-execution self-test. Route matching is pure, so all of it is covered
// here with no server, no socket and no model.
#include <cstdio>
#include <functional>

namespace {

int g_pass = 0;
int g_fail = 0;

void record(const char * unit, const std::string & in, const std::string & out, const std::string & exp) {
    std::printf("[%s] IN : %s\n[%s] OUT: %s\n[%s] EXP: %s\n",
                unit, in.c_str(), unit, out.c_str(), unit, exp.c_str());
    if (out == exp) {
        std::printf("[%s] OK\n", unit);
        ++g_pass;
    } else {
        std::printf("[%s] FAIL\n", unit);
        ++g_fail;
    }
}

void guarded(const char * unit, const std::function<void()> & body) {
    try {
        body();
    } catch (const std::exception & ex) {
        std::printf("[%s] FAIL unexpected throw: %s\n", unit, ex.what());
        ++g_fail;
    }
}

std::string joined(const std::vector<std::string> & parts) {
    std::string out;
    for (const auto & part : parts) {
        if (!out.empty()) {
            out += "|";
        }
        out += part;
    }
    return out;
}

minitts::server::HttpRequest make_request(const char * method, const char * path, const char * query = "") {
    minitts::server::HttpRequest request;
    request.method = method;
    request.path = path;
    request.query = query;
    return request;
}

}  // namespace

int main() {
    using namespace audiocpp::api;
    using minitts::server::HttpResponse;

    // ---- path splitting ----
    guarded("split_path", [] {
        record("split_path/simple", "/v1/jobs", joined(split_path("/v1/jobs")), "v1|jobs");
        record("split_path/trailing_slash", "/v1/jobs/", joined(split_path("/v1/jobs/")), "v1|jobs");
        record("split_path/double_slash", "/v1//jobs", joined(split_path("/v1//jobs")), "v1|jobs");
        record("split_path/root", "/", joined(split_path("/")), "");
        record("split_path/empty", "\"\"", joined(split_path("")), "");
    });

    // ---- matching ----
    guarded("match_route/literal", [] {
        PathParams params;
        record("match/exact", "/health vs /health",
               match_route("/health", "/health", params) ? "match" : "no match", "match");
        record("match/wrong_path", "/health vs /healthz",
               match_route("/health", "/healthz", params) ? "match" : "no match", "no match");
        record("match/too_short", "/v1/jobs vs /v1",
               match_route("/v1/jobs", "/v1", params) ? "match" : "no match", "no match");
        record("match/too_long", "/v1/jobs vs /v1/jobs/x",
               match_route("/v1/jobs", "/v1/jobs/x", params) ? "match" : "no match", "no match");
        record("match/trailing_slash_ok", "/v1/jobs vs /v1/jobs/",
               match_route("/v1/jobs", "/v1/jobs/", params) ? "match" : "no match", "match");
    });

    guarded("match_route/captures", [] {
        PathParams params;
        const bool ok = match_route("/v1/jobs/{id}", "/v1/jobs/job_01ABC", params);
        record("match/one_capture", "/v1/jobs/job_01ABC", ok ? params["id"] : "<no match>", "job_01ABC");

        params.clear();
        const bool two = match_route("/v1/jobs/{id}/outputs/{name}",
                                     "/v1/jobs/job_01ABC/outputs/audio", params);
        record("match/two_captures", "/v1/jobs/job_01ABC/outputs/audio",
               two ? params["id"] + "," + params["name"] : "<no match>", "job_01ABC,audio");

        params.clear();
        record("match/capture_wont_span_slash", "{id} vs two segments",
               match_route("/v1/jobs/{id}", "/v1/jobs/a/b", params) ? "match" : "no match", "no match");

        params.clear();
        // A percent-encoded segment is decoded once, into the capture.
        match_route("/v1/jobs/{id}", "/v1/jobs/a%20b", params);
        record("match/decodes_capture", "/v1/jobs/a%20b", params["id"], "a b");
    });

    // ---- query parsing ----
    guarded("query", [] {
        record("query_value/present", "status=running&limit=5",
               query_value("status=running&limit=5", "status"), "running");
        record("query_value/second", "status=running&limit=5",
               query_value("status=running&limit=5", "limit"), "5");
        record("query_value/absent", "status=running", query_value("status=running", "model"), "");
        record("query_value/empty_query", "\"\"", query_value("", "status"), "");
        record("query_value/no_value", "pinned", query_value("pinned", "pinned"), "");
        record("query_value/decodes", "q=a%20b", query_value("q=a%20b", "q"), "a b");
        record("query_value/plus_is_space", "q=a+b", query_value("q=a+b", "q"), "a b");
        record("query_value/trailing_amp", "a=1&", query_value("a=1&", "a"), "1");
        record("parse_query/count", "a=1&b=2&c=3", std::to_string(parse_query("a=1&b=2&c=3").size()), "3");
    });

    guarded("url_decode", [] {
        record("url_decode/plain", "hello", url_decode("hello"), "hello");
        record("url_decode/percent", "a%2Fb", url_decode("a%2Fb"), "a/b");
        record("url_decode/incomplete", "a%2", url_decode("a%2"), "a%2");
        record("url_decode/not_hex", "a%zz", url_decode("a%zz"), "a%zz");
    });

    // ---- dispatch ----
    guarded("dispatch", [] {
        Router router;
        router.add("GET", "/health", [](const minitts::server::HttpRequest &, const PathParams &) {
            return minitts::server::json_response("{\"ok\":true}");
        });
        router.add("GET", "/v1/jobs", [](const minitts::server::HttpRequest &, const PathParams &) {
            return minitts::server::json_response("{\"list\":true}");
        });
        router.add("POST", "/v1/jobs", [](const minitts::server::HttpRequest &, const PathParams &) {
            return minitts::server::json_response("{\"created\":true}", 202);
        });
        router.add("GET", "/v1/jobs/{id}", [](const minitts::server::HttpRequest &, const PathParams & p) {
            return minitts::server::json_response("{\"id\":\"" + p.at("id") + "\"}");
        });

        record("dispatch/get", "GET /health",
               router.dispatch(make_request("GET", "/health")).body, "{\"ok\":true}");
        record("dispatch/status", "POST /v1/jobs",
               std::to_string(router.dispatch(make_request("POST", "/v1/jobs")).status), "202");
        record("dispatch/same_path_two_methods", "GET /v1/jobs",
               router.dispatch(make_request("GET", "/v1/jobs")).body, "{\"list\":true}");
        record("dispatch/capture_reaches_handler", "GET /v1/jobs/job_9",
               router.dispatch(make_request("GET", "/v1/jobs/job_9")).body, "{\"id\":\"job_9\"}");

        const auto missing = router.dispatch(make_request("GET", "/nope"));
        record("dispatch/unknown_path", "GET /nope", std::to_string(missing.status), "404");
        record("dispatch/404_is_typed", "GET /nope",
               missing.body.find("not_found") != std::string::npos ? "typed" : missing.body, "typed");

        // The distinction worth having: the path is real, the verb is not.
        const auto wrong_verb = router.dispatch(make_request("DELETE", "/v1/jobs"));
        record("dispatch/wrong_method", "DELETE /v1/jobs", std::to_string(wrong_verb.status), "405");
        record("dispatch/allow_header", "DELETE /v1/jobs",
               wrong_verb.headers.count("Allow") ? wrong_verb.headers.at("Allow") : "<absent>",
               "GET, POST");

        record("path_exists/known", "/v1/jobs", router.path_exists("/v1/jobs") ? "yes" : "no", "yes");
        record("path_exists/unknown", "/nope", router.path_exists("/nope") ? "yes" : "no", "no");
        record("methods_for/two", "/v1/jobs", joined(router.methods_for("/v1/jobs")), "GET|POST");
    });

    guarded("dispatch/first_match_wins", [] {
        Router router;
        router.add("GET", "/v1/jobs/queue", [](const minitts::server::HttpRequest &, const PathParams &) {
            return minitts::server::json_response("{\"literal\":true}");
        });
        router.add("GET", "/v1/jobs/{id}", [](const minitts::server::HttpRequest &, const PathParams &) {
            return minitts::server::json_response("{\"capture\":true}");
        });
        // Registration order decides, so a literal route registered first is not
        // shadowed by a capture that would also match.
        record("dispatch/literal_before_capture", "GET /v1/jobs/queue",
               router.dispatch(make_request("GET", "/v1/jobs/queue")).body, "{\"literal\":true}");
        record("dispatch/capture_still_works", "GET /v1/jobs/job_1",
               router.dispatch(make_request("GET", "/v1/jobs/job_1")).body, "{\"capture\":true}");
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_ROUTER
