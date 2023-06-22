var path = require("path");

module.exports = {
    mode: "development",
    entry: "./client.js",
    resolve: {
        modules: [path.resolve(__dirname, "node_modules")],
    },
};
