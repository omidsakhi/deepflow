import { Injectable } from '@angular/core';

declare var io: any;

@Injectable()
export class DataService {
  
  public socket : any = null;

  constructor() { }

  connect() {
    this.socket = io();
    this.socket.on('session_data', (data) => {
      console.log(data);
    })    
  }

}