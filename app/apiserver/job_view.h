#pragma once

// One concern: how a job looks on the wire.
//
// Separate from job.h, which defines how a job looks on disk. They are close
// today but they answer to different audiences: the disk format answers to
// forward compatibility, the wire format to whatever a client needs to render.
// Keeping them apart means adding a convenience field to the API costs nothing
// on disk, and vice versa.

#include "job.h"
#include "job_store.h"

#include <string>
#include <vector>

namespace audiocpp::api {

struct JobViewOptions {
    // The request body can be large (lyrics, a score), so list responses leave it
    // out unless asked.
    bool include_request = true;
    // -1 when the job is not queued.
    int queue_position = -1;
};

engine::io::json::Value job_envelope(const Job & job, const JobViewOptions & options = {});
engine::io::json::Value job_list_envelope(const JobPage & page, bool include_request);

// The result view: artifact descriptors, never inline audio. A four-minute
// stereo track is ~46 MB of WAV and ~61 MB once base64'd, which is not something
// to put in a JSON body by default.
engine::io::json::Value job_result_envelope(const Job & job);

// Content type for an output, from its declared kind. Used by the download route
// so the browser does the right thing with a .wav versus an .abc.
std::string content_type_for(const JobOutput & output);
std::string download_filename(const Job & job, const JobOutput & output);

// Parses the list-query parameters into a JobQuery, rejecting nonsense with a
// message naming the offending parameter.
JobQuery parse_job_query(const std::unordered_map<std::string, std::string> & params);

}  // namespace audiocpp::api
