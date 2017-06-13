import { Injectable, EventEmitter } from '@angular/core';

declare var io: any;

@Injectable()
export class DataService {
  
  public io : any = null;
  public message : EventEmitter<any> = new EventEmitter();

  constructor() { }

  connect() {    
    this.io = io('http://localhost:8080');
    this.io.on('connect', (socket) => {      
      console.log('connection established.');
    });
    this.io.on('sio-output', (message)=> {
      this.message.emit(message);
    })      
  }

}