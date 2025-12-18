;; SPDX-License-Identifier: AGPL-3.0-or-later
;; SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
;;
;; disinfo-nsai-detector - Guix Package Definition
;; Primary package management (fallback: flake.nix)
;;
;; Usage:
;;   guix shell -D -f guix.scm    # Development shell
;;   guix build -f guix.scm       # Build package
;;
;; TODO: After Rust conversion, update to:
;;   (guix build-system cargo)
;;   Add rust toolchain to native-inputs

(use-modules (guix packages)
             (guix gexp)
             (guix git-download)
             (guix build-system gnu)
             ((guix licenses) #:prefix license:)
             (gnu packages base)
             (gnu packages version-control)
             (gnu packages protobuf))

(define-public disinfo-nsai-detector
  (package
    (name "disinfo-nsai-detector")
    (version "0.1.0")
    (source (local-file "." "disinfo-nsai-detector-checkout"
                        #:recursive? #t
                        #:select? (git-predicate ".")))
    ;; NOTE: Placeholder build system until Rust conversion complete
    ;; After conversion: (build-system cargo-build-system)
    (build-system gnu-build-system)
    (native-inputs
     (list git protobuf))
    (synopsis "Disinformation and AI-generated content detection")
    (description
     "A hybrid neural-symbolic system for detecting disinformation and
AI-generated content.  Combines ONNX Runtime for ML inference with
Souffle Datalog for symbolic reasoning, using NATS JetStream for
scalable message processing.  Part of the hyperpolymath RSR ecosystem.")
    (home-page "https://github.com/hyperpolymath/disinfo-nsai-detector")
    (license license:agpl3+)))

;; Return package for guix shell
disinfo-nsai-detector
