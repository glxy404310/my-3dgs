Fixed the optional telemetry check. Null GPU process memory now requires explicit `NOT_MEASURED_UNAVAILABLE`; only Torch allocated/reserved peaks enforce the 4 GiB GPU limit.

New auditor SHA: `47e86e2219b79c1273c9b05941e9043aef706371c995036dce082cc8313e4d07`.

Static checks passed. No data or model computation ran.
