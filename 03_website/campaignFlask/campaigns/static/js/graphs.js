queue()
    .defer(d3.json, "/data")
    .await(makeGraphs);

function makeGraphs(error, recordsJson) {
	
	//Clean data
	var records = recordsJson;
	var dateFormat = d3.time.format("%Y-%m-%d");
	
	records.forEach(function(d) {
		d["date"] = dateFormat.parse(d["date"]);
		d["longitude"] = +d["LNG"];
		d["latitude"] = +d["LAT"];
		d["Rank"] = +d["Rank"];

	});

	//Create a Crossfilter instance
	var ndx = crossfilter(records);

	//Define Dimensions
	var dateDim = ndx.dimension(function(d) { return d["date"]; });
	var clusterDim = ndx.dimension(function(d) { return d["clusterName"]});
	var transactionSizeDim = ndx.dimension(function(d) { return d["transaction_size"]});
	var jobClassDim = ndx.dimension(function(d) { return d["job_class"]});
	var transactionAmtDim = ndx.dimension(function(d) { return d["logDollars"]});
	var priorityDim = ndx.dimension(function(d) { return d["Rank"]});
	var zipcodeDim = ndx.dimension(function (d) { return d["Zipcode"]})	
	var allDim = ndx.dimension(function(d) {return d;});


	//Group Data
	var numRecordsByDate = dateDim.group();
	var transactionSizeGroup = transactionSizeDim.group();
	var jobClassGroup = jobClassDim.group();
	var transactionGroup = transactionAmtDim.group().reduceCount();
	var clusterGroup = clusterDim.group();
	var priorityGroup = priorityDim.group();
	var all = ndx.groupAll();


	//Define values (to be used in charts)
	var minDate = dateDim.bottom(1)[0]["date"];
	var maxDate = dateDim.top(1)[0]["date"];


    //Charts
    var numberRecordsND = dc.numberDisplay("#number-records-nd"); // just the number
	var timeChart = dc.barChart("#time-chart"); //events over time
	var locationChart = dc.rowChart("#location-row-chart"); //demographic clusters
	var histChart = dc.barChart("#histogram-donations-barchart"); //donation size
	var jobClassChart = dc.rowChart("#jobclass-row-chart"); //job class
	var donationChart = dc.rowChart("#donation-row-chart"); //

	
	
	 histChart
 		.width(300)
 		.height(150)
 		.x(d3.scale.linear().domain([0,10]))
 		.dimension(transactionAmtDim) 
 		.brushOn(false)
     	.group(transactionGroup)
     	.colors(['#6baed6'])
     	.elasticX(true)
		.elasticY(true)
		.xAxisLabel('log(donation size) ($)')
		.xAxis().ticks(4);	

	numberRecordsND
		.formatNumber(d3.format("d"))
		.valueAccessor(function(d){return d; })
		.group(all);


	timeChart
		.width(650)
		.height(140)
		.margins({top: 10, right: 50, bottom: 20, left: 20})
		.dimension(dateDim)
		.group(numRecordsByDate)
		.transitionDuration(500)
		.x(d3.time.scale().domain([minDate, maxDate]))
		.elasticY(true)
		.yAxis().ticks(4);
	
	jobClassChart
		.width(300)
		.height(300)
		.dimension(jobClassDim)
		.group(jobClassGroup)
		.ordering(function(d) { return -d.value })
		.colors(['#6baed6'])
		.elasticX(true)
		.xAxis().ticks(4);	
	
		
	donationChart //Donation Size
		.width(300)
		.height(150)
		.dimension(transactionSizeDim)
		.group(transactionSizeGroup)
		.ordering(function(d) { return -d.value})
		.colors(['#6baed6'])
        .elasticX(true)
        .xAxis().ticks(4);
	
    locationChart // clusters
    	.width(200)
		.height(510)
        .dimension(clusterDim)
        .group(clusterGroup)
        .ordering(function(d) { return -d.value })
        .colors(['#6baed6'])
        .elasticX(true)
        .labelOffsetY(10)
        .xAxis().ticks(4);	


    var map = L.map('map');
	
	var drawMap = function(){

	    map.setView([43, -92], 3);
		mapLink = '<a href="http://openstreetmap.org">OpenStreetMap</a>';
		L.tileLayer(
			'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
				attribution: '&copy; ' + mapLink + ' Contributors',
				maxZoom: 15,
			}).addTo(map);

		//HeatMap
		var geoData = [];
		_.each(allDim.top(Infinity), function (d) {
			geoData.push([d["latitude"], d["longitude"], 1]);
	      });
		var heat = L.heatLayer(geoData,{
			radius: 5,
			blur: 7, 
			maxZoom: 5,
		}).addTo(map);
		
		// add markers to heatmap
		var markerData = [];
		_.each(priorityDim.top(20), function (d) {
			markerData.push([d["latitude"], d["longitude"], d["diff"], d["Zipcode"]]);
	      });

		for (var i = 0; i < markerData.length; i++) {
			marker = new L.marker([markerData[i][0],markerData[i][1]])
				.bindPopup(markerData[i][3] + ": $" + Math.round(markerData[i][2],2))
				.addTo(map);
		}

	};

	//Draw Map
	drawMap();

	dcCharts = [timeChart, locationChart, jobClassChart, donationChart, histChart]; 

	_.each(dcCharts, function (dcChart) {
		dcChart.on("filtered", function (chart, filter) {
			map.eachLayer(function (layer) {
				map.removeLayer(layer)
			}); 
			drawMap();
		});
	});

	dc.renderAll();

};

