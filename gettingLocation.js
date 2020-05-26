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

let inputElement = document.getElementById('FileUpload');

//получение массива из выбранного файла
inputElement.onchange = function(event) {
   var fileList = inputElement.files;
   parseData(fileList[0]);
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
  let dataForMap = latLongArray;
  let testData = {
     max: 20,
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
    "radius": 2,
    "maxOpacity": .8,
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
    layers: [baseLayer, heatmapLayer]
    });

    heatmapLayer.setData(testData);
};
