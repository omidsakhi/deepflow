//var express = require('express')
//var app = express()
//var http = require('http').Server(app);
//var io = require('socket.io')(http);
var io = require('socket.io')(8080);

//app.use(express.static('public'))

//app.get('/', function (req, res) {
//	res.sendfile('index.html');
//});

var sockets = [];

io.on('connection', function (socket) {	
	var index = sockets.indexOf(socket);
	if (index == -1) {		
		sockets.push(socket);
		console.log('a new connection added.');
	}
	socket.on('sio-output', function (data) {				
		for (var i = 0; i < sockets.length; i++) {
			if (sockets[i] != socket) {				
				sockets[i].emit('sio-output',data);
			}
		}
	})
	socket.on('disconnect', function () {		
		var index = sockets.indexOf(socket);
		if (index != -1) {			
			sockets.splice(index, 1);
			console.log('a connection lost.');
		}
	});
});

console.log('http://localhost:8080/socket.io/socket.io.js');

//http.listen(3000, function () {
//	console.log('listening on *:3000');
//});

