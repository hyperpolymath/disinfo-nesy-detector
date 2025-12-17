;;; STATE.scm - Project Checkpoint
;;; disinfo-nsai-detector
;;; Format: Guile Scheme S-expressions
;;; Purpose: Preserve AI conversation context across sessions
;;; Reference: https://github.com/hyperpolymath/state.scm

;; SPDX-License-Identifier: AGPL-3.0-or-later
;; SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell

;;;============================================================================
;;; METADATA
;;;============================================================================

(define metadata
  '((version . "0.1.0")
    (schema-version . "1.0")
    (created . "2025-12-15")
    (updated . "2025-12-17")
    (project . "disinfo-nsai-detector")
    (repo . "github.com/hyperpolymath/disinfo-nsai-detector")))

;;;============================================================================
;;; PROJECT CONTEXT
;;;============================================================================

(define project-context
  '((name . "disinfo-nsai-detector")
    (tagline . "Disinformation and AI-generated content detection")
    (version . "0.1.0")
    (license . "AGPL-3.0-or-later")
    (rsr-compliance . "gold-target")

    (tech-stack
     ((current . "Go (pending Rust conversion)")
      (target . "Rust (async-nats, ort, prometheus, prost)")
      (ci-cd . "GitHub Actions + GitLab CI + Bitbucket Pipelines")
      (security . "CodeQL + OSSF Scorecard + TruffleHog")
      (package-mgmt . "Guix (primary) + Nix (fallback)")))))

;;;============================================================================
;;; CURRENT POSITION
;;;============================================================================

(define current-position
  '((phase . "v0.1 - Initial Setup and RSR Compliance")
    (overall-completion . 25)

    (components
     ((rsr-compliance
       ((status . "complete")
        (completion . 100)
        (notes . "SHA-pinned actions, SPDX headers, multi-platform CI")))

      (documentation
       ((status . "foundation")
        (completion . 30)
        (notes . "README exists, META/ECOSYSTEM/STATE.scm added")))

      (testing
       ((status . "minimal")
        (completion . 10)
        (notes . "CI/CD scaffolding exists, limited test coverage")))

      (core-functionality
       ((status . "in-progress")
        (completion . 25)
        (notes . "Initial implementation underway")))))

    (working-features
     ("RSR-compliant CI/CD pipeline"
      "Multi-platform mirroring (GitHub, GitLab, Bitbucket)"
      "SPDX license headers on all files"
      "SHA-pinned GitHub Actions"))))

;;;============================================================================
;;; ROUTE TO MVP
;;;============================================================================

(define route-to-mvp
  '((target-version . "1.0.0")
    (definition . "Stable release with comprehensive documentation and tests")

    (milestones
     ((v0.2
       ((name . "Core Functionality")
        (status . "pending")
        (items
         ("Implement primary features"
          "Add comprehensive tests"
          "Improve documentation"))))

      (v0.5
       ((name . "Feature Complete")
        (status . "pending")
        (items
         ("All planned features implemented"
          "Test coverage > 70%"
          "API stability"))))

      (v1.0
       ((name . "Production Release")
        (status . "pending")
        (items
         ("Comprehensive test coverage"
          "Performance optimization"
          "Security audit"
          "User documentation complete"))))))))

;;;============================================================================
;;; BLOCKERS & ISSUES
;;;============================================================================

(define blockers-and-issues
  '((critical
     ())  ;; No critical blockers

    (high-priority
     ((rust-conversion
       ((description . "Go code must be converted to Rust per RSR policy")
        (impact . "Non-compliant with RSR language requirements")
        (needed . "Convert cmd/main.go and pkg/* to Rust")
        (reference . "RUST_CONVERSION_NEEDED.md")))))

    (medium-priority
     ((test-coverage
       ((description . "Limited test infrastructure")
        (impact . "Risk of regressions")
        (needed . "Comprehensive test suites")))
      (cargo-toml
       ((description . "Missing Cargo.toml for Rust build")
        (impact . "Cannot build Rust code")
        (needed . "Create Cargo.toml with dependencies")))))

    (low-priority
     ((documentation-gaps
       ((description . "Some documentation areas incomplete")
        (impact . "Harder for new contributors")
        (needed . "Expand documentation")))))))

;;;============================================================================
;;; CRITICAL NEXT ACTIONS
;;;============================================================================

(define critical-next-actions
  '((immediate
     (("Create Cargo.toml with Rust dependencies" . high)
      ("Convert cmd/main.go to Rust with tokio+async-nats" . high)
      ("Convert pkg/onnx_wrapper to Rust with ort crate" . high)))

    (this-week
     (("Convert pkg/souffle_wrapper to Rust" . high)
      ("Generate Rust protobuf bindings with prost" . medium)
      ("Add initial Rust test coverage" . medium)))

    (this-month
     (("Complete Go -> Rust conversion" . critical)
      ("Update guix.scm to cargo-build-system" . medium)
      ("Reach v0.2 milestone" . high)))))

;;;============================================================================
;;; SESSION HISTORY
;;;============================================================================

(define session-history
  '((snapshots
     ((date . "2025-12-17")
      (session . "scm-security-review")
      (accomplishments
       ("Fixed RSR_COMPLIANCE.adoc with accurate status"
        "Fixed META.scm syntax error (cross-platform-status)"
        "Created flake.nix for Nix fallback package management"
        "Updated STATE.scm with Rust conversion priorities"
        "Reviewed all 11 GitHub security workflows - all SHA-pinned"))
      (notes . "Security and SCM review session"))

     ((date . "2025-12-15")
      (session . "initial-state-creation")
      (accomplishments
       ("Added META.scm, ECOSYSTEM.scm, STATE.scm"
        "Established RSR compliance"
        "Created initial project checkpoint"))
      (notes . "First STATE.scm checkpoint created via automated script")))))

;;;============================================================================
;;; HELPER FUNCTIONS (for Guile evaluation)
;;;============================================================================

(define (get-completion-percentage component)
  "Get completion percentage for a component"
  (let ((comp (assoc component (cdr (assoc 'components current-position)))))
    (if comp
        (cdr (assoc 'completion (cdr comp)))
        #f)))

(define (get-blockers priority)
  "Get blockers by priority level"
  (cdr (assoc priority blockers-and-issues)))

(define (get-milestone version)
  "Get milestone details by version"
  (assoc version (cdr (assoc 'milestones route-to-mvp))))

;;;============================================================================
;;; EXPORT SUMMARY
;;;============================================================================

(define state-summary
  '((project . "disinfo-nsai-detector")
    (version . "0.1.0")
    (overall-completion . 30)
    (next-milestone . "v0.2 - Rust Conversion + Core Functionality")
    (critical-blockers . 0)
    (high-priority-issues . 1)
    (primary-blocker . "Go -> Rust conversion required")
    (updated . "2025-12-17")))

;;; End of STATE.scm
