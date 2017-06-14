var port = 6643;
var io = require('socket.io')(port);

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

console.log('http://localhost:' + port + '/socket.io/socket.io.js');

