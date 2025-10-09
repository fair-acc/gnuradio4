// Simple HTTP server used only for unit tests.
// Interactive use (from the command line): node fileio_pre.js start
// Automated use (in unit tests): Call startServer() in unit test after embedding this file into the JS runtime via --pre-js.
// See: https://emscripten.org/docs/tools_reference/emcc.html#emcc-pr


var http = require("http");
// this requires xhr2 to be installed, please run: npm install xhr2
// see: https://github.com/emscripten-core/emscripten/issues/21158

XMLHttpRequest = require('xhr2');

let   getNumbersAndErrorCounter = 0;
const numbersString = "0123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";

const server = http.createServer((req, res) => {
    console.log(req.url);
    if (req.url == "/echo") {
        res.writeHead(200, {"Content-Type" : "text/plain"});
        res.write("Hello world!");
        res.end();
    } else if (req.url == "/getNumbers") {
        res.writeHead(200, {"Content-Type" : "text/plain"});
        res.write(numbersString);
        res.end();
    } else if (req.url == "/getNumbersTimeout1s") {
        return setTimeout(() => {
            res.writeHead(200, {"Content-Type" : "text/plain"});
            res.write(numbersString);
            res.end();
        }, 1000);
    } else if (req.url == "/getNumbersAndError") {
        getNumbersAndErrorCounter++; // long polling with 2 OK responses and then error
        if (getNumbersAndErrorCounter <= 2) {
            return setTimeout(() => {
                res.writeHead(200, {"Content-Type" : "text/plain"});
                res.write(numbersString);
                res.end();
            }, 100);
        } else {
            return setTimeout(() => {
                res.writeHead(500, {"Content-Type" : "text/plain"});
                res.write("Simulated server error after 2 events");
                res.end();
            }, 100);
        }
    } else if (req.url == "/postNumbers") {
        let body = "";
        req.on("data", (chunk) => { body += chunk; });
        req.on("end", () => {
            res.writeHead(200, {"Content-Type" : "text/plain"});
            if (body === numbersString) {
                res.write("OK");
            } else {
                res.write("ERROR");
            }
            res.end();
        });
    } else {
        res.writeHead(404, {"Content-Type" : "text/plain"});
        res.write("Unknown");
        res.end();
    }
});

var startServer = () => {
    getNumbersAndErrorCounter = 0;
    server.once("error", (err) => { console.log(err); });
    server.listen(8080);
};

var stopServer = () => {
    server.close();
    server.closeAllConnections();
};

if (process.argv.at(-1) == "start") {
    startServer();
} else if (process.argv.at(-1) == "stop") {
    stopServer();
}
