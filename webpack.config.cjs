const path = require("path");

module.exports = {
  mode: "production",
  entry: "./src/core/index.ts",
  experiments: {
    outputModule: true,
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "tinysam.js",
    library: {
      type: "module",
    },
    publicPath: "",
    assetModuleFilename: "[name].[contenthash][ext]",
  },
  resolve: {
    extensions: [".ts", ".js"],
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.onnx$/,
        type: "asset/inline",
        generator: {
          dataUrl: {
            encoding: "base64",
            mimetype: "application/octet-stream",
          },
        },
      },
    ],
  },
  externals: {
    "onnxruntime-web": "onnxruntime-web",
  },
  externalsType: "module",
  devtool: "source-map",
};
