// this is a little HTTP server intended for unit test purposes.
// for interactive CLI usage, you can start the server by passing "start" as the last argument, e.g.:
// node pre.js start
//
// for automated usage in unit tests, please call startServer() yourself after embedding this into the JS runtime with --pre-js
// see: https://emscripten.org/docs/tools_reference/emcc.html#emcc-pre-js

var http = require("http");
// this requires xhr2 to be installed, please run: npm install xhr2
// see: https://github.com/emscripten-core/emscripten/issues/21158
/*global XMLHttpRequest:writable*/
XMLHttpRequest = require('xhr2');

const server = http.createServer((req, res) => {
    console.log(req.url);
    if (req.url == "/echo") {
        res.writeHead(200, {"Content-Type": "text/plain"});
        res.write("Hello world!");
        res.end();
    } else if (req.url == "/notify") {
        // wait a bit before responding (we are testing long polling here)
        return setTimeout(() => {
            res.writeHead(200, {"Content-Type": "text/plain"});
            res.write("event");
            res.end();
        }, 10);
    } else if (req.url == "/number") {
        res.writeHead(200, {"Content-Type": "text/plain"});
        res.write("OK");
        res.end();
    } else {
        res.writeHead(404, {"Content-Type": "text/plain"});
        res.write("Unknown route");
        res.end();
    }
});

var startServer = () => {
    server.once("error", (err) => { console.log(err); });
    server.listen(8080);
    console.log("Started the server");
};

var stopServer = () => {
    server.close();
    server.closeAllConnections();
    console.log("Stopped the server");
};

if (process.argv.at(-1) == "start") {
    startServer();
} else if (process.argv.at(-1) == "stop") {
    stopServer();
}
