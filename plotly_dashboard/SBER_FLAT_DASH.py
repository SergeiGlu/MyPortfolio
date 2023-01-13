# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import json
import pickle

import catboost as cb


<<<<<<< HEAD
df_train = pd.read_csv(Path(Path.cwd(), 'plotly_dashboard/data', 'train.csv'), parse_dates=['timestamp'])
=======

df_train = pd.read_csv(Path(Path.cwd(), 'data', 'train.csv'), parse_dates=['timestamp'])
>>>>>>> f8fd6b3b33120f075f5b71aac91d1eb3f66266d1


## Импорт файла geojson
def load_geojson():
    data_path_geojson = Path(Path.cwd(), 'plotly_dashboard/data', 'mo.geojson')
    with open(data_path_geojson) as f:
        data_geojson = json.load(f)
    
    return data_geojson

## импорт модели
def load_model():
    ## Find the path
    data_path = Path(Path.cwd(), 'plotly_dashboard/model', 'model_cb_dash.pkl')

    ## Load model
    with open(data_path, "rb") as f:
        model = pickle.load(f)
    
    return model

## Словарь для районов (латиница-кириллица)
df_area_dict = pd.read_csv(Path(Path.cwd(), 'plotly_dashboard/data', 'merge_area.csv'), sep = ';')

## Создаем список для передачи в карту
list_propert = list(set([x['properties']['NAME'] for x in load_geojson()['features']]))
df_spis = (pd.DataFrame({'area_2':list_propert}).merge(### добавляем словарь районов
                                                        (df_train.merge(df_area_dict, left_on = 'sub_area', right_on = 'area_1')
                                                                .rename(columns = {'area_2':'area_rus'})
                                                        [['sub_area','area_rus']].drop_duplicates()
                                                        )
            , left_on = 'area_2', right_on = 'area_rus', how = 'left'
            ).rename(columns = {'area_rus':'flag_exist_area'})
            [['area_2','flag_exist_area']]
            )

## Флаг существования области в train выборки
df_spis['flag_exist_area'] = df_spis['flag_exist_area'].fillna('Нет').apply(lambda x: 'Да' if x != 'Нет' else 'Нет')

## Словарь для поиска включен район в модель или нет
dict_df_spis = df_spis.set_index('area_2').T.to_dict('records')[0]

## Создаем карту
fig = px.choropleth(df_spis, geojson=load_geojson()
                            , locations="area_2"
                            , featureidkey="properties.NAME"
                            , color="flag_exist_area"
                            , color_discrete_map={"Нет": 'red', "Да": 'green'}
                            , hover_name = "area_2"
                            ,labels={'flag_exist_area': 'Есть данные для модели'}
                    )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                  plot_bgcolor =  '#082255',
                  paper_bgcolor = '#082255',
                  font_color = '#fff',
                    )

app = Dash(__name__, external_stylesheets =[dbc.themes.BOOTSTRAP])




#####################################
#########       LAYOUT       ########
#####################################


app.layout = html.Div(children=[
    ### Название ###
    html.Div([
        html.Div([
            html.H2(children='Покупка квартиры', className="app__header__title")
            ,html.Br()
            ,html.H4(children='Dashboard для предсказание цены квартиры', className="app__header__title--grey")
        ], className="app__header__desc"
        )
    ], className="app__header") #, className="app__content"
    ,html.Div([
            html.Div([
                        html.Div([html.H5("КАРТА ОБЛАСТЕЙ", className="graph__title")])
                        ,html.H6("Выберите на карте округ", className="graph__subtitle", style = {'padding':'5px 25px 0px 25px'})
                        ,dcc.Graph(figure = fig, id = 'map_region')
                        ,html.Div(id='choose-region', className="answer__title") #, 'width':'49%' , style = {'float':'left'}
            ], className="wind__speed__container", style = {'width':'52%'})#, style={'display':'inline-block',})
            ,html.Div([
                html.Div([html.H5("ФИЛЬТРЫ", className="graph__title")], style={'margin-bottom':'10px'})
                ,html.H6("", className="graph__subtitle", style = {'padding':'5px 25px 0px 25px'})
                ,html.Div([
                    html.Div([
                        html.H6("Покупка/Инвестиция", className="graph__subtitle")
                        ,dcc.Dropdown(
                                    options=[
                                        {'label':'Покупка', 'value':'OwnerOccupier'},
                                        {'label':'Инвестиция', 'value':'Investment'}
                                    ],
                                    value = 'OwnerOccupier',
                                    id = "prodtype"
                        )
                        ,html.H6("Месяц", className="graph__subtitle")
                        ,dcc.Input(id='month_output',
                                type='number',
                                value = 5,
                                min = 0,
                                max = 12
                        )
                        ,html.H6("Год", className="graph__subtitle")
                        ,dcc.Input(id='year_output',
                                type='number',
                                value = 2011,
                                max = 2030,
                                min = 2000
                        )
                        ,html.H6("Этаж", className="graph__subtitle")
                        ,dcc.Input(id='floor_output',
                                type='number',
                                value = 3,
                                min = 0
                        )
                        ,html.H6("Количество комнат", className="graph__subtitle")
                        ,dcc.Input(id='num_room_output',
                                type='number',
                                value = 2,
                                min = 0
                        )
                        ,html.H6("Состояние квартиры", className="graph__subtitle")
                        ,dcc.Input(id='state_output',
                                type='number',
                                value = 1,
                                min = 0,
                                max = 4
                        )
                        ,html.H6("Общая площадь квартиры", className="graph__subtitle")
                        ,dcc.Input(id='full_sq',
                                type='number',
                                value = 43,
                                min = 0
                        )
                        ,html.H6("Жилая площадь квартиры", className="graph__subtitle")
                        ,dcc.Input(id='lifesq_output',
                                type='number',
                                value = 27,
                                min = 0
                        )
                        ], className='block__filter')
                    ,html.Div([
                        html.H6("Доступность к учебным заведениям (км)", className="graph__subtitle")
                        ,dcc.Input(id='kindergarten_km_output',
                                type='number',
                                value = 0.15,
                                min = 0,
                        )
                        ,html.H6("Доступность к учреждениям дошкольного образования (км)", className="graph__subtitle")
                        ,dcc.Input(id='preschool_km',
                                type='number',
                                value = 0.17,
                                min = 0,
                        )
                        ,html.H6("Доступность к учреждениям дополнительного образования (км)", className="graph__subtitle")
                        ,dcc.Input(id='additional_education_km',
                                type='number',
                                value = 0.95,
                                min = 0,
                        )
                        ,html.H6("Количество спортивных сооружений в районе 3 км", className="graph__subtitle")
                        ,dcc.Input(id='sport_count_3000_output',
                                type='number',
                                value = 21,
                                min = 0,
                        )
                        ,html.H6("Расстояние до ЖД станции (км)", className="graph__subtitle")
                        ,dcc.Input(id='railroad_station_avto_km_output',
                                type='number',
                                value = 5.419,
                                min = 0,
                        )
                        ,html.H6("Расстояние до больницы", className="graph__subtitle")
                        ,dcc.Input(id='public_healthcare_km_output',
                                type='number',
                                value = 0.97,
                                min = 0,
                        )
                        ,html.H6("Расстояние до метро", className="graph__subtitle")
                        ,dcc.Input(id='metro_km_avto_output',
                                type='number',
                                value = 1.13,
                                min = 0,
                        )
                        ,html.H6("Расстояние до электростаниции", className="graph__subtitle")
                        ,dcc.Input(id='ts_km_output',
                                type='number',
                                value = 4.3,
                                min = 0,
                        )
                        ], className='block__filter')
                ], style={'display':'flex','flex-direction':'row', 'justify-content': 'space-between'})
            ,dbc.Button("Применить", outline=True, color="primary", className="block_button", id = 'filter_button')
            ], className="wind__speed__container", style = {'width':'47%'}
            )
    ], className = "block__container")
    ,html.Div([
        html.Div([html.H6("Предсказанная стоимость", className="graph__subtitle")])
        ,html.Div(id='output__answer', className="answer__title")
    ], className = "wind__speed__container", style = {'margin-top':'10px'})
    ,html.H6('Данные: ', className="block__link",style = {'padding-top':'25px'})
    ,html.Div([
        html.A(children='Sberbank Russian Housing Market'
                ,href='https://www.kaggle.com/competitions/sberbank-russian-housing-market'
                )
    ],className="block__link")
])


@app.callback(
    Output(component_id='choose-region', component_property = 'children'),
    Input(component_id = 'map_region',component_property='clickData')
)

def get_value_figure(clickData):
    if clickData is None:
        return "Округ не выбран"
    else:
        # print(clickData)
        location = clickData['points'][0]['location']

        if dict_df_spis[location] == 'Да':
            return f'Выбранный округ: {location}'
        else:
            return 'Нет данных для этого округа'


# Сделать кнопку
@app.callback(
    Output(component_id="output__answer", component_property='children'),
    [Input(component_id= 'filter_button', component_property='n_clicks')],
    [State(component_id= 'map_region', component_property='clickData'),
    State(component_id= 'prodtype', component_property='value'),
    State(component_id= 'month_output', component_property='value'),
    State(component_id= 'year_output', component_property='value'),
    State(component_id= 'floor_output', component_property='value'),
    State(component_id= 'num_room_output', component_property='value'),
    State(component_id= 'state_output', component_property='value'),
    State(component_id= 'lifesq_output', component_property='value'),
    State(component_id= 'kindergarten_km_output', component_property='value'),
    State(component_id= 'sport_count_3000_output', component_property='value'),
    State(component_id= 'railroad_station_avto_km_output', component_property='value'),
    State(component_id= 'public_healthcare_km_output', component_property='value'),
    State(component_id= 'metro_km_avto_output', component_property='value'),
    State(component_id= 'ts_km_output', component_property='value'),
    State(component_id= 'full_sq', component_property='value'),
    State(component_id= 'additional_education_km', component_property='value'),
    State(component_id= 'preschool_km', component_property='value')
    ]
)

def get_value_model(n_clicks, map_region, prodtype, month_output,
                    year_output, floor_output, num_room_output
                    ,state_output, lifesq_output, kindergarten_km_output
                    ,sport_count_3000_output, railroad_station_avto_km_output
                    ,public_healthcare_km_output, metro_km_avto_output, ts_km_output
                    ,full_sq, additional_education_km, preschool_km
                    ):

    location = get_value_figure(map_region)
    if (prodtype is None
    or location == "" or location == 'Округ не выбран'
    or month_output is None
    or year_output is None
    or floor_output is None
    or num_room_output is None
    or state_output is None
    or lifesq_output is None
    or kindergarten_km_output is None
    or sport_count_3000_output is None
    or railroad_station_avto_km_output is None
    or public_healthcare_km_output is None
    or metro_km_avto_output is None
    or ts_km_output is None
    or full_sq is None
    or additional_education_km is None
    or preschool_km is None
    ):
        return 'Выберите все фильтры'
    elif location == 'Нет данных для этого округа':
        return 'Выберете правильный округ'
    else:

        ### Загружаем модель ###
        model = load_model()

        ### Формируем Data ###

        data = ({
        'full_sq': [full_sq],
        'year' : [year_output],
        'floor' : [floor_output],
        'num_room' : [num_room_output],
        'state' : [state_output],
        'preschool_km' : [preschool_km], 
        'product_type' : [prodtype],
        'life_sq'      : [lifesq_output],
        'additional_education_km' : [additional_education_km],
        'kindergarten_km' : [kindergarten_km_output],
        'sport_count_3000' : [sport_count_3000_output],
        'month' : [month_output],
        'railroad_station_avto_km' : [railroad_station_avto_km_output],
        'public_healthcare_km' : [public_healthcare_km_output],
        'metro_km_avto': [metro_km_avto_output],
        'ts_km': [ts_km_output],
        'area_2': [location.replace('Выбранный округ: ','')]
        })

        df_data = pd.DataFrame.from_dict(data)

        ## Обработка входящего массива
        col_categ = [6, 16]
        df_data_cat = cb.Pool(df_data, cat_features=col_categ)
        
        pred = np.exp(model.predict(df_data_cat))

        return '{:,.2f} руб.'.format(float(pred)).replace(',', ' ')

if __name__ == '__main__':
    app.run_server(debug=True)