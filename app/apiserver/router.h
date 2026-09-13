#pragma once

// One concern: turning a method and a path into a handler call, with the
// variable path segments extracted.
//
// A table rather than the if/else chain the existing server grew: routes here
// carry captures (/v1/jobs/{id}/outputs/{name}), and matching those by hand is
// exactly the kind of code that quietly accepts a path it should not.

#include "http.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace audiocpp::api {

// Captured path segments, keyed by the name in the pattern.
using PathParams = std::unordered_map<std::string, std::string>;

using RouteHandler = std::function<minitts::server::HttpResponse(const minitts::server::HttpRequest &,
                                                                 const PathParams &)>;

struct Route {
    std::string method;
    std::string pattern;   // e.g. "/v1/jobs/{id}/outputs/{name}"
    RouteHandler handler;
};

// Splits a path on '/', dropping empty segments so a trailing slash does not
// change the match.
std::vector<std::string> split_path(const std::string & path);

// Matches one pattern against one path, filling `params`. A "{name}" segment
// captures; every other segment must match exactly.
bool match_route(const std::string & pattern, const std::string & path, PathParams & params);

// One query parameter, url-decoded. Empty when absent.
std::string query_value(const std::string & query, const std::string & key);
std::unordered_map<std::string, std::string> parse_query(const std::string & query);
std::string url_decode(const std::string & value);

class Router {
public:
    void add(std::string method, std::string pattern, RouteHandler handler);

    // Dispatches, returning 404 for an unknown path and 405 (with an Allow
    // header) when the path exists but not for this method -- a distinction the
    // existing server collapses into 404, and one that saves a client guessing.
    minitts::server::HttpResponse dispatch(const minitts::server::HttpRequest & request) const;

    // True when any route matches this path, whatever the method.
    bool path_exists(const std::string & path) const;
    std::vector<std::string> methods_for(const std::string & path) const;

private:
    std::vector<Route> routes_;
};

}  // namespace audiocpp::api
