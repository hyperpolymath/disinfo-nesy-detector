;; disinfo-nsai-detector - Guix Package Definition
;; Run: guix shell -D -f guix.scm

(use-modules (guix packages)
             (guix gexp)
             (guix git-download)
             (guix build-system gnu)
             ((guix licenses) #:prefix license:)
             (gnu packages base))

(define-public disinfo_nsai_detector
  (package
    (name "disinfo-nsai-detector")
    (version "0.1.0")
    (source (local-file "." "disinfo-nsai-detector-checkout"
                        #:recursive? #t
                        #:select? (git-predicate ".")))
    (build-system gnu-build-system)
    (synopsis "Guix channel/infrastructure")
    (description "Guix channel/infrastructure - part of the RSR ecosystem.")
    (home-page "https://github.com/hyperpolymath/disinfo-nsai-detector")
    (license license:agpl3+)))

;; Return package for guix shell
disinfo_nsai_detector
