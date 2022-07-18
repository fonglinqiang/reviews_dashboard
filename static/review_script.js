// import { WordCloudController, WordElement } from '../node_modules/chartjs-chart-wordcloud/build/index.umd';

// Chart.register(WordCloudController, WordElement);

chartIt();

async function chartIt(){
    
    console.log(data)
    var as_of_date = data['as_of_date']
    var chart_country = data['chart_country']
    var chart_pie = data['chart_pie']
    var chart_ratingbydate = data['chart_ratingbydate']
    var chart_histpolarity = data['chart_histpolarity']
    var wordcloud = data['wordcloud']
    var chart_topics = data['chart_topics']
    var paragraph_issues = data['paragraph_issues']
    var paragraph_features = data['paragraph_features']

    var color_rating1 = '#D5FFD9'
    var color_rating2 = '#A4DAC4'
    var color_rating3 = '#7BB4AE'
    var color_rating4 = '#5B8F94'
    var color_rating5 = '#436B78'
    var color_polarity = '#00722D'
    var color_android = '#3C4A3E'
    var color_ios = '#68AAC5'


    document.getElementById("as_of_date").innerHTML = as_of_date;

    // 1 chart_country
    try {
        var chart_country_android = chart_country['Android']
        var chart_country_country = Object.keys(chart_country_android)
        var chart_country_android_count = Object.values(chart_country_android)
    }
    catch (err) {
        var err_os = 'Android'
    }
    try {
        var chart_country_ios = chart_country['iOS']
        var chart_country_country = Object.keys(chart_country_ios)
        var chart_country_ios_count = Object.values(chart_country_ios)
    }
    catch (err) {
        var err_os = 'iOS'
    }
    console.log(err_os)

    if (err_os == 'Android') {
        var chart_country_country = Object.keys(chart_country_ios)
        var chart_country_android_count = new Array(10).fill(0);
    } else if (err_os == 'iOS') {
        var chart_country_country = Object.keys(chart_country_android)
        var chart_country_ios_count = new Array(10).fill(0);
    }

    // Chart.defaults.font.size = 20

    var ctx = document.getElementById('chart6');
    var config = {
        type: 'bar',
        data: {
            labels: chart_country_country,
            datasets: [
                {
                label: 'Android',
                data: chart_country_android_count,
                backgroundColor: color_android,
                },
                {
                label: 'iOS',
                data: chart_country_ios_count,
                backgroundColor: color_ios,
                },
            ]
        },
        options: {
            plugins: {
                legend: {
                    position: 'right'
                },
                title: {
                    display:true,
                    text: 'Reviews by Language/Country',
                    padding: {top:10,bottom:10},
                    fontSize: 100,
                },
            },
            scales: {
                yAxes: {
                    title: {
                        display:true,
                        text: 'Count'
                    }
                },
                xAxes: {
                    title: {
                        display:true,
                        text: 'Language/Country'
                    }
                }
            }
        },
    };
    var chart1 = new Chart(ctx,config);

    // 2 chart_piebyrating
    var chart_pie = chart_pie['Rating'];
    var chart_pie_label = Object.keys(chart_pie);
    var chart_pie_values = Object.values(chart_pie);

    var ctx2 = document.getElementById('chart1');
    var config2 = {
        type: 'doughnut',
        data: { 
            labels: chart_pie_label,
            datasets: [{
                data: chart_pie_values,
                backgroundColor: [color_rating1,color_rating2,color_rating3,color_rating4,color_rating5],
            }]
        },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: 'Ratings',
                    padding: {top:10,bottom:10}
                },
                legend: {
                    display: false,
                    position: 'bottom'
                }
            }
        }
    };

    var myChart2 = new Chart(ctx2, config2);

    // 3 chart_ratingbydate
    var chart_ratingbydate_labels = Object.keys(chart_ratingbydate['1']);
    var chart_ratingbydate_1 = Object.values(chart_ratingbydate['1']);
    var chart_ratingbydate_2 = Object.values(chart_ratingbydate['2']);
    var chart_ratingbydate_3 = Object.values(chart_ratingbydate['3']);
    var chart_ratingbydate_4 = Object.values(chart_ratingbydate['4']);
    var chart_ratingbydate_5 = Object.values(chart_ratingbydate['5']);

    var ctx3 = document.getElementById('chart2');
    var config3 = {
        type: 'bar',
        data: {
            labels: chart_ratingbydate_labels,
            datasets: [
                {
                    label: '1',
                    data: chart_ratingbydate_1,
                    backgroundColor: color_rating1,
                },
                {
                    label: '2',
                    data: chart_ratingbydate_2,
                    backgroundColor: color_rating2,
                },
                {
                    label: '3',
                    data: chart_ratingbydate_3,
                    backgroundColor: color_rating3,
                },
                {
                    label: '4',
                    data: chart_ratingbydate_4,
                    backgroundColor: color_rating4,
                },
                {
                    label: '5',
                    data: chart_ratingbydate_5,
                    backgroundColor: color_rating5,
                },
            ]
        },
        options: {
            element: {
                bar: {
                    borderwidth:0,
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Ratings from '+as_of_date
                },
                legend: {
                    position: 'right'
                }
            },
            responsive: true,
            scales: {
                x: {
                    stacked: true,
                },
                y: {
                    stacked: true
                }
            }
        }
    };

    var chart3 = new Chart(ctx3, config3);

    // 4 chart_histpolarity
    var chart_histpolarity_labels = Object.keys(chart_histpolarity['polarity']);
    var chart_histpolarity_values = Object.values(chart_histpolarity['polarity']);

    var ctx4 = document.getElementById('chart5');
    var config4 = {
        type: 'bar',
        data: {
            labels: chart_histpolarity_labels,
            datasets: [
                {
                    data:chart_histpolarity_values,
                    backgroundColor: color_polarity,
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display:false,
                    position: 'bottom',
                },
                title: {
                    display: true,
                    text: 'Polarity'
                }
            },
            scales: {
                y: {
                    title: {
                        display:true,
                        text: 'Count'
                    }
                }
            }
        }
    }

    var chart4 = new Chart(ctx4, config4);

    // 5 wordcloud
    var wordcloud_labels = Object.keys(wordcloud);
    var wordcloud_values = Object.values(wordcloud);

    var ctx5 = document.getElementById('chart4');
    var config5 = {
        type: 'wordCloud',
        data: {
            labels:  wordcloud_labels,
            datasets: [
                {
                    label: '',
                    data: wordcloud_values,
                },
            ],
        },
        options: {
            title: {
                display:false,
                text: 'wordcloud'
            },
            plugins: {
                legend: {
                    display:false,
                }
            }
        },
    };
    var chart5 = new Chart(ctx5, config5);


    // 6 chart_topics
    var chart_topics_labels = chart_topics['label'];
    var chart_topics_1 = Object.values(chart_topics['1']);
    var chart_topics_2 = Object.values(chart_topics['2']);
    var chart_topics_3 = Object.values(chart_topics['3']);
    var chart_topics_4 = Object.values(chart_topics['4']);
    var chart_topics_5 = Object.values(chart_topics['5']);

    var ctx6 = document.getElementById('chart3');
    var config6 = {
        type: 'bar',
        data: {
            labels: chart_topics_labels,
            datasets: [
                {
                    label: '1',
                    data: chart_topics_1,
                    backgroundColor: color_rating1,
                },
                {
                    label: '2',
                    data: chart_topics_2,
                    backgroundColor: color_rating2,
                },
                {
                    label: '3',
                    data: chart_topics_3,
                    backgroundColor: color_rating3,
                },
                {
                    label: '4',
                    data: chart_topics_4,
                    backgroundColor: color_rating4,
                },
                {
                    label: '5',
                    data: chart_topics_5,
                    backgroundColor: color_rating5,
                },
            ]
        },
        options: {
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: 'Features Ratings (Count)'
                },
                legend: {
                    position: 'bottom'
                }
            },
            responsive: true,
            scales: {
                x: {
                    stacked: true,
                    max:100
                },
                y: {
                    stacked: true
                }
            }
        }
    };

    var chart6 = new Chart(ctx6, config6);

    // 7 paragraph_issues
    var paragraph_issues_ad = paragraph_issues['ADVERTISEMENT']
    var paragraph_issues_bugs = paragraph_issues['BUGS']
    var paragraph_issues_others = paragraph_issues['OTHERS']
    var paragraph_issues_privacy = paragraph_issues['PRIVACY']

    document.getElementById("issues_ad").innerHTML = paragraph_issues_ad;
    document.getElementById("issues_bugs").innerHTML = paragraph_issues_bugs;
    document.getElementById("issues_privacy").innerHTML = paragraph_issues_privacy;
    document.getElementById("issues_others").innerHTML = paragraph_issues_others;

    // 8 paragraph_features
    document.getElementById("feature1").innerHTML = paragraph_features["0"]["topic"].toUpperCase();
    document.getElementById("rating1").innerHTML = paragraph_features["0"]["Rating"];
    document.getElementById("review1").innerHTML = paragraph_features["0"]["translated"];
    document.getElementById("feature2").innerHTML = paragraph_features["1"]["topic"].toUpperCase();
    document.getElementById("rating2").innerHTML = paragraph_features["1"]["Rating"];
    document.getElementById("review2").innerHTML = paragraph_features["1"]["translated"];
    document.getElementById("feature3").innerHTML = paragraph_features["2"]["topic"].toUpperCase();
    document.getElementById("rating3").innerHTML = paragraph_features["2"]["Rating"];
    document.getElementById("review3").innerHTML = paragraph_features["2"]["translated"];
    document.getElementById("feature4").innerHTML = paragraph_features["3"]["topic"].toUpperCase();
    document.getElementById("rating4").innerHTML = paragraph_features["3"]["Rating"];
    document.getElementById("review4").innerHTML = paragraph_features["3"]["translated"];
    document.getElementById("feature5").innerHTML = paragraph_features["4"]["topic"].toUpperCase();
    document.getElementById("rating5").innerHTML = paragraph_features["4"]["Rating"];
    document.getElementById("review5").innerHTML = paragraph_features["4"]["translated"];
    document.getElementById("feature6").innerHTML = paragraph_features["5"]["topic"].toUpperCase();
    document.getElementById("rating6").innerHTML = paragraph_features["5"]["Rating"];
    document.getElementById("review6").innerHTML = paragraph_features["5"]["translated"];
    document.getElementById("feature7").innerHTML = paragraph_features["6"]["topic"].toUpperCase();
    document.getElementById("rating7").innerHTML = paragraph_features["6"]["Rating"];
    document.getElementById("review7").innerHTML = paragraph_features["6"]["translated"];
    document.getElementById("feature8").innerHTML = paragraph_features["7"]["topic"].toUpperCase();
    document.getElementById("rating8").innerHTML = paragraph_features["7"]["Rating"];
    document.getElementById("review8").innerHTML = paragraph_features["7"]["translated"];
    document.getElementById("feature9").innerHTML = paragraph_features["8"]["topic"].toUpperCase();
    document.getElementById("rating9").innerHTML = paragraph_features["8"]["Rating"];
    document.getElementById("review9").innerHTML = paragraph_features["8"]["translated"];
    document.getElementById("feature10").innerHTML = paragraph_features["9"]["topic"].toUpperCase();
    document.getElementById("rating10").innerHTML = paragraph_features["9"]["Rating"];
    document.getElementById("review10").innerHTML = paragraph_features["9"]["translated"];

}

