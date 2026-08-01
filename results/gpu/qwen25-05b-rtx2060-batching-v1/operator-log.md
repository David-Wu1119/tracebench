# Operator Log

All timestamps are UTC. The registered implementation commit was
`885f9209a80cce3b5f230a3b59a0a8a4a5cd8e75`; the plan was committed before the
first output-producing run and has SHA-256
`d813bd11bf60d9845438d9865b4592e426adeab7686a923a06b7409ad1da44cc`.

## Run chronology

### Valid run 1, excluded from the clean replication pair

- Started: `2026-08-01T05:53:36.080389Z`
- Completed: `2026-08-01T06:02:38.423827Z`
- Outcome: 352 completed records, zero failed records; verified result rows match
  the later clean runs.
- Raw archive SHA-256:
  `110d83f98e35fed9f880ba864ac4d51c0b325bf090ca00fb781f29c99ced9f28`

The 542.343-second wall time was roughly three times the later clean-run duration,
and continuous process observation had not yet been established. The evidence
bundle is contract-valid, but it is excluded from the clean replication pair
because desktop GPU contention could not be ruled out.

### Aborted attempt 1

- Started: `2026-08-01T06:07:15.154125Z`
- Model loaded: `2026-08-01T06:07:18.677400Z`
- Greedy mode completed: `2026-08-01T06:13:48.708348Z`
- Sampled mode started and the process was then terminated.
- Partial execution rows retained: 208 of 352.

Steam/Apex processes had restarted and were observed competing for the GPU. The
operator terminated the experiment rather than allow it to produce a final
evidence directory. The `.partial` directory has no final manifest or checksum
sidecar and is not included in any metric.

Private partial archive SHA-256:
`47f4352bc51a2a94cae9b166335a2971ee0d7d44f4985c11dff69dacda4d685d`.

### Aborted attempt 2

- Started: `2026-08-01T06:16:34.125221Z`
- Model loaded: `2026-08-01T06:16:35.999627Z`
- Greedy mode completed: `2026-08-01T06:18:18.545747Z`
- Sampled mode started and the process was then terminated.
- Partial execution rows retained: 183 of 352.

Steam/Apex restarted again. The operator terminated the process, retained the
`.partial` directory, and excluded it from every metric.

Private partial archive SHA-256:
`5ef4e64317ff1b090115d67e0a8a829c7642037d2b815c8b34b0d61e90a7d30a`.

The contention attribution in both aborted attempts is an operator observation;
the partial evidence contract does not independently bind a continuous process
trace. That is why the attempts are disclosed as invalid operator events rather
than analyzed experimental runs.

### Clean run 2

- Started: `2026-08-01T06:20:44.226275Z`
- Completed: `2026-08-01T06:23:42.479277Z`
- Wall time: 178.253 seconds
- Outcome: 352 completed records, zero failed records
- Executions SHA-256:
  `2b68523a4bea293e28b073659de55fb70f78a44264535bfd9a156b9bd81e2ac3`
- Raw archive SHA-256:
  `516f2f387ba34e4fbe87254e8d5cd09ea2ba2af325c949ecf868db9ae229a31f`

Before this run, the full Steam process tree and the game were stopped. GPU process
checks during the run showed the experiment and ordinary desktop processes, with
no Steam/Apex process observed.

### Clean run 3

- Started: `2026-08-01T06:27:08.308050Z`
- Completed: `2026-08-01T06:30:06.941327Z`
- Wall time: 178.633 seconds
- Outcome: 352 completed records, zero failed records
- Executions SHA-256:
  `2b97e563ea9d8d58b41830783a9f0ef018f1cc6c13b799c0341c5e9988acf5f9`
- Raw archive SHA-256:
  `86150dfc831c04eed449ba7ac9dfd244b4f54018db1984f84ce6a8200e935259`

The GPU process list was monitored during this run. No Steam/Apex process was
observed.

## Replication decision

Clean runs 2 and 3 are the reported replication pair. Their independent manifests
have different timestamps and execution-file hashes. The public comparator
verified the same plan, runtime digest, implementation commit, and exact 352-key
execution set, then found zero status, output-digest, or error mismatches.

Run 1 is retained but excluded conservatively. Both aborted attempts are retained
as private partial archives and disclosed here; neither has a complete evidence
contract, and neither enters a denominator.
