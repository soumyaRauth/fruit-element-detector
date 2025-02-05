import type { NextConfig } from "next";
import type { Configuration } from 'webpack';
import webpack from 'webpack';

const nextConfig: NextConfig = {
  webpack: (config: Configuration) => {
    config.plugins = config.plugins || [];  // Initialize if undefined
    config.plugins.push(
      new webpack.IgnorePlugin({
        checkResource: (resource) => {
          return /node-pre-gyp[\\/]+lib[\\/]+util[\\/]+nw-pre-gyp[\\/]+index\.html$/.test(resource);
        }
      })
    );
    // Rule for .html files, but exclude node_modules to avoid processing dependency HTML files
    config.module = config.module || {}; // Initialize module if undefined
    config.module.rules = config.module.rules || []; // Initialize rules if undefined
    config.module.rules.push({
      test: /\.html$/,
      exclude: /node_modules/,
      type: 'asset/source'
    });

    return config;
  }
};

export default nextConfig;
