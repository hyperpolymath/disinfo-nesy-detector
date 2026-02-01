; SPDX-License-Identifier: PMPL-1.0-or-later
; NEUROSYM.scm - Neurosymbolic context for disinfo-nesy-detector
; Media type: application/vnd.neurosym+scm

(neurosym
  (metadata
    (version "1.0.0")
    (schema-version "1.0")
    (created "2026-01-30")
    (updated "2026-01-30"))

  (conceptual-model
    (domain "ai-security")
    (subdomain "automation")
    (core-concepts
      (concept "tool"
        (definition "A software component that automates tasks")
        (properties "input" "output" "configuration"))))

  (knowledge-graph-hints
    (entities "disinfo-nesy-detector" "Rust" "automation")
    (relationships
      ("disinfo-nesy-detector" provides "automation-capabilities"))))
