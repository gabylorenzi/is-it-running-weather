let button = document.querySelector('.button')
let city_name = document.querySelector('.name')
let temperature = document.querySelector('.temp')
let wind_speed = document.querySelector('.wind')
let inputvalue = document.querySelector('.inputValue')

button.addEventListener('click', function(){
	fetch('api.openweathermap.org/data/2.5/weather?q=${inputvalue.value}&appid=29dabdd9237ba71d01601da27f3d052b')
		.then(response => response.json())
		.then(makeitpretty)
})

const makeitpretty=(weather)=>{
	city_name.innerText = `${weather.name}`
	temp.innerText = `${weather.main.temp}`
	wind_speed.innerText = `${weather.wind.speed}`
}