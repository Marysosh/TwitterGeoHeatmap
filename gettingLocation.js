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
           let smallArr=dataArray.slice(63,74);
           console.log(smallArr);
           getPageName(smallArr);
 	        }
      });
}

let inputElement = document.getElementById('FileUpload');

//получение массива из выбранного файла
inputElement.onchange = function(event) {
   var fileList = inputElement.files;
   parseData(fileList[0]);
}

//отправление никнеймов на сервер и получение координат
function getPageName(NicknameArray){
  for (let i=0; i<NicknameArray.length; i++){
    let Server = 'http://127.0.0.1:3000/getCoord?nickname=' + NicknameArray[i];
    fetch(Server)
      .then(response => response.json())
      .then(result => console.log(result))
  }
}
