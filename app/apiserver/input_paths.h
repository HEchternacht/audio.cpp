#pragma once

// One concern: keeping a client's filenames inside the job they belong to.
//
// The CLI resolves a path by handing an absolute one straight through and
// joining a relative one to the working directory. That is right for a CLI --
// the operator owns the machine and means what they typed. Reached over HTTP it
// is an arbitrary file read: a request naming `C:/Users/.../id_rsa` is opened
// and fed to a model, and `../../` walks out of the job's staging directory
// just as easily. The server binds a loopback port with `cors_origins: "*"`, so
// any page in the browser can post one.
//
// So every path a client supplies is confined here before the request is built:
// a plain name, inside the job's own input directory, and nothing else. Uploads
// are copied in under names this accepts, which is what makes the rule easy to
// keep rather than something each family has to remember.

#include "engine/framework/io/json.h"

#include <string>

namespace audiocpp::api {

// True when `value` is a plain relative name that stays inside its base
// directory: not absolute, no drive or root, and no `..` component.
bool is_confined_name(const std::string & value);

// The name an upload is stored and staged under. Keeps the extension, which is
// how audio decoding picks a container, and replaces anything else that could
// steer a path.
std::string safe_upload_name(const std::string & requested);

// Throws `bad_request` for any path-shaped field in a submitted request body
// that would escape the job's input directory. Mutates nothing.
void confine_request_paths(const engine::io::json::Value & request);

}  // namespace audiocpp::api
