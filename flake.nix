{
  inputs.nixpkgs.url = "github:NixOs/nixpkgs/nixpkgs-unstable";
  inputs.zig.url = "github:mitchellh/zig-overlay";

  outputs = {
    self,
    nixpkgs,
    ...
  } @ inputs: let
    system = "x86_64-linux";
    overlays = [
      (final: prev: {zig = inputs.zig.packages.${prev.system}."0.15.2";})
    ];
    pkgs = import nixpkgs {
      overlays = overlays;
      system = system;
      config.allowUnfree = true;
      config.cudaSupport = true;
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = with pkgs; [
        zig
        lldb
        zls
        # cudaPackages.cudatoolkit
        (with pkgs.python3Packages; [
          scikit-learn
          polars
          numpy
          matplotlib
          plotnine
          ipython
        ])
      ];

      # CUDA_PATH = "${pkgs.cudatoolkit}";
      # shellHook = ''
      #   export PATH="/run/opengl-driver/lib:$PATH"
      # '';
    };
  };
}
