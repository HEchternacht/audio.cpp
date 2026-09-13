#include "api_error.h"

#include "busy_guard.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/run_control.h"

namespace audiocpp::api {

ApiError bad_request(const std::string & message) { return ApiError(400, "invalid_request_error", message); }
ApiError not_found(const std::string & message) { return ApiError(404, "not_found", message); }
ApiError conflict(const std::string & message) { return ApiError(409, "conflict", message); }
ApiError busy(const std::string & message) { return ApiError(503, "server_busy", message); }
ApiError internal(const std::string & message) { return ApiError(500, "server_error", message); }

minitts::server::HttpResponse error_response(int status, const std::string & message, const std::string & type) {
    const std::string body = "{\"error\":{\"message\":" + engine::io::json::stringify_string(message) +
                             ",\"type\":" + engine::io::json::stringify_string(type) + "}}";
    return minitts::server::json_response(body, status);
}

minitts::server::HttpResponse error_response(const ApiError & error) {
    return error_response(error.status(), error.what(), error.type());
}

minitts::server::HttpResponse guard(const std::function<minitts::server::HttpResponse()> & body) {
    try {
        return body();
    } catch (const ApiError & ex) {
        return error_response(ex);
    } catch (const minitts::server::InsufficientMemoryError & ex) {
        return error_response(503, ex.what(), "insufficient_memory");
    } catch (const minitts::server::ServerBusyError & ex) {
        return error_response(503, ex.what(), "server_busy");
    } catch (const engine::runtime::RunCancelled &) {
        return error_response(409, "the run was cancelled", "conflict");
    } catch (const std::exception & ex) {
        // Anything unmapped is a server fault, not the caller's: say 500 rather
        // than letting the exception escape and drop the connection.
        return error_response(500, ex.what(), "server_error");
    }
}

}  // namespace audiocpp::api

#ifdef AUDIOCPP_SELFTEST_API_ERROR
#include <cstdio>

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

}  // namespace

int main() {
    using namespace audiocpp::api;

    guarded("constructors", [] {
        record("bad_request/status", "message", std::to_string(bad_request("x").status()), "400");
        record("bad_request/type", "message", bad_request("x").type(), "invalid_request_error");
        record("not_found/status", "message", std::to_string(not_found("x").status()), "404");
        record("not_found/type", "message", not_found("x").type(), "not_found");
        record("conflict/status", "message", std::to_string(conflict("x").status()), "409");
        record("conflict/type", "message", conflict("x").type(), "conflict");
        record("busy/status", "message", std::to_string(busy("x").status()), "503");
        record("internal/status", "message", std::to_string(internal("x").status()), "500");
        record("message_preserved", "the model exploded", bad_request("the model exploded").what(),
               "the model exploded");
    });

    guarded("error_response/shape", [] {
        const auto response = error_response(404, "unknown model 'x'", "not_found");
        record("error_response/status", "404", std::to_string(response.status), "404");
        record("error_response/content_type", "json", response.content_type, "application/json");
        record("error_response/body", "unknown model 'x'", response.body,
               "{\"error\":{\"message\":\"unknown model 'x'\",\"type\":\"not_found\"}}");
    });

    guarded("error_response/escapes", [] {
        // A message carrying a quote or a newline must not break the envelope.
        const auto response = error_response(400, "bad \"input\"\nline two", "invalid_request_error");
        record("error_response/escaping", "quotes and a newline",
               response.body.find("bad \\\"input\\\"\\nline two") != std::string::npos ? "escaped"
                                                                                       : response.body,
               "escaped");
    });

    guarded("guard/passthrough", [] {
        const auto response = guard([] { return minitts::server::json_response("{\"ok\":true}"); });
        record("guard/no_throw", "handler returns", response.body, "{\"ok\":true}");
    });

    guarded("guard/maps_api_error", [] {
        const auto response = guard([]() -> minitts::server::HttpResponse {
            throw not_found("no such job");
        });
        record("guard/api_error_status", "throws not_found", std::to_string(response.status), "404");
        record("guard/api_error_message", "throws not_found",
               response.body.find("no such job") != std::string::npos ? "carried" : response.body, "carried");
    });

    guarded("guard/maps_busy_guard_errors", [] {
        // These come from the shared busy guard and already mean 503 elsewhere;
        // the API must not reclassify them.
        const auto busy_response = guard([]() -> minitts::server::HttpResponse {
            throw minitts::server::ServerBusyError("model is busy");
        });
        record("guard/server_busy", "ServerBusyError", std::to_string(busy_response.status), "503");
        record("guard/server_busy_type", "ServerBusyError",
               busy_response.body.find("server_busy") != std::string::npos ? "typed" : busy_response.body,
               "typed");

        const auto memory_response = guard([]() -> minitts::server::HttpResponse {
            throw minitts::server::InsufficientMemoryError("not enough VRAM");
        });
        record("guard/insufficient_memory", "InsufficientMemoryError",
               std::to_string(memory_response.status), "503");
        record("guard/insufficient_memory_type", "InsufficientMemoryError",
               memory_response.body.find("insufficient_memory") != std::string::npos ? "typed"
                                                                                     : memory_response.body,
               "typed");
    });

    guarded("guard/maps_unknown", [] {
        const auto response = guard([]() -> minitts::server::HttpResponse {
            throw std::runtime_error("something unexpected");
        });
        record("guard/unknown_status", "std::runtime_error", std::to_string(response.status), "500");
        record("guard/unknown_type", "std::runtime_error",
               response.body.find("server_error") != std::string::npos ? "typed" : response.body, "typed");
    });

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
#endif  // AUDIOCPP_SELFTEST_API_ERROR
