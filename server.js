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
        res.json({status:'ok', geo: geolocations})
    })
    .catch(function (err) {
        console.log('ERROR!',err);
        res.json({status:'error'})
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
