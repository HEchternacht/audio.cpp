#pragma once

// One concern: the error vocabulary, and turning a thrown error into a response.
//
// The shape matches what audiocpp_server already returns --
// {"error":{"message":..,"type":..}} -- so a client written against one works
// against the other. One status code is added: 409, for a request that was
// well-formed but arrived in the wrong state (cancelling a finished job,
// deleting a running one). Reporting those as 400 would tell a client to fix a
// request that is not wrong.

#include "http.h"

#include <functional>
#include <stdexcept>
#include <string>

namespace audiocpp::api {

class ApiError : public std::runtime_error {
public:
    ApiError(int status, std::string type, const std::string & message)
        : std::runtime_error(message), status_(status), type_(std::move(type)) {}

    int status() const noexcept { return status_; }
    const std::string & type() const noexcept { return type_; }

private:
    int status_;
    std::string type_;
};

// The five cases the API actually raises. Free functions rather than subclasses:
// there is no behaviour to override, only a status and a type to carry.
ApiError bad_request(const std::string & message);
ApiError not_found(const std::string & message);
ApiError conflict(const std::string & message);
ApiError busy(const std::string & message);
ApiError internal(const std::string & message);

minitts::server::HttpResponse error_response(int status, const std::string & message, const std::string & type);
minitts::server::HttpResponse error_response(const ApiError & error);

// Runs `body`, mapping anything it throws onto a response. ApiError keeps its
// status; the busy-guard errors keep the 503s they already mean elsewhere;
// anything else is a 500 rather than an escaped exception killing a connection.
minitts::server::HttpResponse guard(const std::function<minitts::server::HttpResponse()> & body);

}  // namespace audiocpp::api
