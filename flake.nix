# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
#
# disinfo-nsai-detector - Nix Flake (fallback package management)
# Primary: guix.scm | Fallback: flake.nix
{
  description = "Disinformation and AI-generated content detection system";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Rust toolchain for future Rust conversion
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # Development dependencies
        buildInputs = with pkgs; [
          # Rust toolchain (for conversion)
          rustToolchain

          # Build tools
          just
          protobuf

          # Runtime dependencies
          nats-server

          # Security scanning
          trufflehog

          # Container tools
          podman
          podman-compose
        ];

        nativeBuildInputs = with pkgs; [
          pkg-config
          openssl
        ];

      in
      {
        # Development shell
        devShells.default = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;

          shellHook = ''
            echo "disinfo-nsai-detector development environment"
            echo "Primary package management: Guix (guix.scm)"
            echo "This is the Nix fallback environment"
            echo ""
            echo "Available commands:"
            echo "  just        - Run development tasks"
            echo "  cargo       - Rust build (after conversion)"
            echo "  nats-server - Start NATS for testing"
            echo ""
            echo "NOTE: Go -> Rust conversion required (see RUST_CONVERSION_NEEDED.md)"
          '';

          RUST_BACKTRACE = "1";
          RUST_LOG = "debug";
        };

        # Package definition (placeholder until Rust conversion)
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "disinfo-nsai-detector";
          version = "0.1.0";

          src = ./.;

          nativeBuildInputs = nativeBuildInputs;
          buildInputs = buildInputs;

          buildPhase = ''
            echo "Build requires Rust conversion - see RUST_CONVERSION_NEEDED.md"
            echo "Current: Go placeholder"
          '';

          installPhase = ''
            mkdir -p $out/bin
            echo "#!/bin/sh" > $out/bin/disinfo-nsai-detector
            echo "echo 'Rust conversion required - see RUST_CONVERSION_NEEDED.md'" >> $out/bin/disinfo-nsai-detector
            chmod +x $out/bin/disinfo-nsai-detector
          '';

          meta = with pkgs.lib; {
            description = "Disinformation and AI-generated content detection";
            homepage = "https://github.com/hyperpolymath/disinfo-nsai-detector";
            license = licenses.agpl3Plus;
            maintainers = [ ];
            platforms = platforms.linux ++ platforms.darwin;
          };
        };

        # Formatter
        formatter = pkgs.nixpkgs-fmt;
      });
}
