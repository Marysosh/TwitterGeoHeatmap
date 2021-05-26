var dataArray = [];//массив с полученными из файла данными

//функция, которая парсит данные из csv файла в массив
 function parseData(file) {
      Papa.parse(file, {
        worker: true,
        step: function(results) {
          dataArray.push(results.data[1]);
         },
         complete: function()
 	       {
 		       console.log("Row:", dataArray);
           let smallArr=dataArray.slice(1,5000);
           console.log(smallArr);
           let promise = getPageName(smallArr);
           promise.then( result => {

             mapConstructor(result);
           })
 	        }
      });
}

 function onReaderLoad(event){
	var obj = JSON.parse(event.target.result);
	mapConstructor(obj);
}

let inputElement = document.getElementById('FileUpload');

//получение массива из выбранного файла
inputElement.onchange = function(event) {
   // var fileList = inputElement.files;
   // parseData(fileList[0]);
   
   var reader = new FileReader();
	reader.onload = onReaderLoad;
	reader.readAsText(event.target.files[0]);
}

//получение координат пользователя
function getPageName(NicknameArray) {
  return new Promise(async function(resolve, reject){
    let usersLocationsArray = [];
    let coordinates = [];
    let latLongArray = [];
    let coordinatesNumbers = {};
    let maxCount = -10;
    for (let i=0; i<NicknameArray.length; i++){
      let Server = 'http://127.0.0.1:3000/getCoord?nickname=' + NicknameArray[i];
      await fetch(Server)
        .then(response => response.json())
        .then(result => {
          usersLocationsArray[i] = result;
          usersLocationsArray[i].nickname = NicknameArray[i];
          console.log(result);
          if (usersLocationsArray[i].coordinates){
            coordinates[i] = usersLocationsArray[i].coordinates;


          if ((String(coordinates[i][0]) +', '+ String(coordinates[i][1])) in coordinatesNumbers){
            coordinatesNumbers[String(coordinates[i][0]) +', '+ String(coordinates[i][1])] += 1;
          }else{
            coordinatesNumbers[String(coordinates[i][0]) +', '+ String(coordinates[i][1])] =1;
          };

          let j = 0;
          for (let key in coordinatesNumbers){
            let latLong = key.split(', ');
            latLongArray[j] = new Object();
            latLongArray[j].lat = parseFloat(latLong[1]);
            latLongArray[j].long = parseFloat(latLong[0]);
            latLongArray[j].count = coordinatesNumbers[key];
            j+=1;
            if (coordinatesNumbers[key] > maxCount){
              maxCount = coordinatesNumbers[key];
            }
          };
        };
        })
    };
    console.log(usersLocationsArray);
    console.log(coordinates);
    console.log(coordinatesNumbers);
    console.log(latLongArray);

    resolve(latLongArray);
    reject(latLongArray)
  });
};

//построение карты
function mapConstructor(latLongArray){
  let dataForMap = latLongArray.locations;
  console.log(latLongArray);
  let testData = {
     //max: Math.max.apply(Math, dataForMap.map(function(o) { return o.count; })),
     data: dataForMap
   };

   let baseLayer = L.tileLayer(
   'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
      attribution: '...',
      maxZoom: 18
    }
    );

    let cfg = {
    // radius should be small ONLY if scaleRadius is true (or small radius is intended)
    // if scaleRadius is false it will be the constant radius used in pixels
    "radius": 1,
	"blur": 0.95,
    "maxOpacity": .6,
	//"minOpacity": .3,
	//"gradient": {0.001: 'blue', 0.05: 'lime', 0.3: 'yellow', 0.5: 'red'},
	"gradient": {.01:"MediumSlateBlue",.1:"cyan",.2:"lime",.3:"yellow",.6:"red"},
	zoomDelta: 0.1,
    // scales the radius based on map zoom
    "scaleRadius": true,
    // if set to false the heatmap uses the global maximum for colorization
    // if activated: uses the data maximum within the current map boundaries
    //   (there will always be a red spot with useLocalExtremas true)
    "useLocalExtrema": true,
    // which field name in your data represents the latitude - default "lat"
    latField: 'lat',
    // which field name in your data represents the longitude - default "lng"
    lngField: 'long',
    // which field name in your data represents the data value - default "value"
    valueField: 'count'
    };

    let heatmapLayer = new HeatmapOverlay(cfg);

    let map = new L.Map('map', {
    center: new L.LatLng(25.6586, -80.3568),
    zoom: 4,
	minZoom: 0,
		maxZoom: 18,
		zoomSnap: 0,
		zoomDelta: 0.1,
    layers: [baseLayer, heatmapLayer]
    });

	var cartodbAttribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://carto.com/attribution">CARTO</a>';

	var positron = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', {
		attribution: cartodbAttribution
	}).addTo(map);

	var ZoomViewer = L.Control.extend({
		
		zoomDelta: 0.1,
		onAdd: function(){

			var container= L.DomUtil.create('div');
			var gauge = L.DomUtil.create('div');
			container.style.width = '200px';
			container.style.background = 'rgba(255,255,255,0.5)';
			container.style.textAlign = 'left';
			map.on('zoomstart zoom zoomend', function(ev){
				gauge.innerHTML = 'Zoom level: ' + map.getZoom();
			})
			container.appendChild(gauge);

			return container;
		}
	});

	(new ZoomViewer).addTo(map);

	map.setView([0, 0], 0);


    heatmapLayer.setData(testData);
};
