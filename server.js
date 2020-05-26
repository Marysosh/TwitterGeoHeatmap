const path = require('path');
const express = require('express');
const rp = require('request-promise')
const app = express();
const port = 3000;

//прописывание пути до статических файлов(JS, CSS)
app.use(express.static(path.join(__dirname, '')));

//при переходе на адрес / вывод страницы page.html
app.get('/', (request, response) => {
    response.sendFile(path.join(__dirname+'/page.html'));
});

//отправляем запрос от клиента, получаем код страницы пользователя, находим геопозицию пользователя
app.get('/getCoord', (req,res) => {
  nickname = req.query.nickname;
  //response.json({data: nickname})
  const options = {
    method: 'GET',
    uri: 'https://twitter.com/'+ nickname
  };
  console.log(options);
  rp(options)
    .then(function (response) {
        //console.log(response)
         let pageCode = response;
         let geo = /ProfileHeaderCard-locationText u-dir" dir="ltr">([^<]+)</g.exec(pageCode);
         let geolocations = geo[1].trim();
         console.log(geolocations);
        //res.json({status:'successfully'});
  // получение координат через API OpenStreetMap
        if (geolocations) {
              const options = {
                method: 'GET',
                uri: 'https://nominatim.openstreetmap.org/search?q=' + geolocations + '&limit=49&format=geojson',
                headers:{
                'User-Agent': 'Request-Promise'
              },
              json: true
              };
              //console.log(options);
              rp(options)
              .then(function(response) {
                let streetMapInfo = response;
                if (streetMapInfo.features[0]){
                  let coordinates = streetMapInfo.features[0].geometry.coordinates
                  console.log(coordinates);
                  res.json({status:'Coordinates found!', geolocation: geolocations, coordinates: coordinates })
                } else {
                  console.log('No coordinates from OpenStreetMap');
                  res.json({status:'no coordinates from OpenStreetMap', geolocation: geolocations, scoordinates:''})
                };
              })

        } else {
          console.log('No geolocation from profile');
          res.json({status: 'No geolocation from users page tag', coordinates:''});
        };
        //res.json({status:'ok', geo: geolocations})
    })
    .catch(function (err) {
        console.log('No twitter page code');
        res.json({status:'No twitter page code', coordinates:''})
    });
});

//слушаем порт и принимаем запросы
app.listen(port, (err) => {
    if (err) {
        return console.log('something bad happened', err);
    };
    console.log(`server is listening on ${port}`);
});


// работает let geo = pageCode.match(/ProfileHeaderCard-locationText u-dir" dir="ltr">[^<]+</g);
//<span class="ProfileHeaderCard-locationText u-dir" dir="ltr">The Internet, USA</span>
