import { Component, AfterViewInit } from '@angular/core';
import { DataService } from './data.service';

declare var d3: any;

export class LossData {
  constructor(
    public dtrue: number,
    public dfake: number,
    public gfake: number,
    public iter: number
  ) { }
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent implements AfterViewInit {

  data: LossData[] = [];
  count = 0;

  margin = { top: 20, right: 20, bottom: 30, left: 40 };
  svg: any = null;
  g: any = null;
  width = 0;
  height = 0;
  xTransform: any = null;
  yTransform: any = null;
  xAxis: any = null;
  yAxis: any = null;
  gFakeLine: any = null;
  gfake = 0;
  dfake = 0;
  dtrue = 0;
  gFakeColor = "#85929E";
  dTrueColor = "#D7BDE2";
  dFakeColor = "#F7DC6F";

  constructor(private dataService: DataService) {
    dataService.connect();
    dataService.message.subscribe((message) => {
      this.processData(message);
    });
  }

  ngAfterViewInit() {
    const that = this;

    this.svg = d3.select("svg");
    this.width = this.svg.attr("width") - this.margin.left - this.margin.right;
    this.height = this.svg.attr("height") - this.margin.top - this.margin.bottom;
    this.xTransform = d3.scaleLinear().range([0, this.width]).domain([0, 1]);
    this.yTransform = d3.scaleLinear().range([this.height, 0]).domain([0, 100]);
    this.xAxis = d3.axisBottom(this.xTransform);
    this.yAxis = d3.axisLeft(this.yTransform);

    this.svg.append("defs").append("clipPath")
      .attr("id", "clip")
      .append("rect")
      .attr("width", this.width)
      .attr("height", this.height);

    this.g = this.svg.append("g")
      .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")");

    this.g.append("g")
      .attr("class", "axis axis--x")
      .attr("transform", "translate(0," + this.height + ")")
      .call(this.xAxis);

    this.g.append("g")
      .attr("class", "axis axis--y")
      .call(this.yAxis);

    this.g.append("path")
      .attr("fill", this.gFakeColor)
      .attr("opacity", 0.6)
      .attr("class", "gfake");

    this.g.append("path")
      .attr("fill", this.dFakeColor)
      .attr("opacity", 0.6)
      .attr("class", "dfake");

    this.g.append("path")
      .attr("fill", this.dTrueColor)
      .attr("opacity", 0.6)
      .attr("class", "dtrue");

    this.update();
  }

  update() {

    this.xTransform.domain([0, this.count]);

    var gFakeArea = d3.area()
      .curve(d3.curveMonotoneX)
      .x((d) => { return this.xTransform(d.iter); })
      .y0(this.height)
      .y1((d) => { return this.yTransform(d.gfake); });

    var dFakeArea = d3.area()
      .curve(d3.curveMonotoneX)
      .x((d) => { return this.xTransform(d.iter); })
      .y0(this.height)
      .y1((d) => { return this.yTransform(d.dfake); });

    var dTrueArea = d3.area()
      .curve(d3.curveMonotoneX)
      .x((d) => { return this.xTransform(d.iter); })
      .y0(this.height)
      .y1((d) => { return this.yTransform(d.dtrue); });

    this.g.select('.gfake')
      .attr('d', gFakeArea(this.data));
    this.g.select('.dfake')
      .attr('d', dFakeArea(this.data));
    this.g.select('.dtrue')
      .attr('d', dTrueArea(this.data));

    this.g.select('.axis--x')
      .call(this.xAxis);
  }

  round(n: number) {
    return parseFloat('' + (Math.round(n * 100) / 100)).toFixed(2);
  }

  processData(message: any) {


    for (let i = 0; i < message.inputs.length; i++) {
      const input = message.inputs[i];
      if (input.name === 'dtrue') {
        this.dtrue = input.data[0];
      } else if (input.name === 'dfake') {
        this.dfake = input.data[0];
      } else if (input.name === 'gfake') {
        this.gfake = input.data[0];
      }
    }

    this.data.push(new LossData(this.dtrue, this.dfake, this.gfake, this.count));
    this.count++;
    this.update();
  }

}
