import os

import psycopg2

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_auth
import plotly.graph_objs as go
import pandas as pd
from pandas import read_sql_query
from sqlalchemy import create_engine
# import sqlalchemy
import datetime
import numpy as np
from dash.dependencies import Input, Output
from colour import Color


# Uncomment for production.

# Create database connection
DATABASE_URL = os.environ['DATABASE_URL']
conn = psycopg2.connect(DATABASE_URL, sslmode='require')

# Comment out for production.
# conn = create_engine('postgresql://postgres:isldatabase@localhost/ISL_pgre')

config = {'displayModeBar': False}

white = Color("#dcdcdc")
colors = list(white.range_to(Color("blue"),1000))

country_choice_options = [
    {'label': 'All countries', 'value': 'ALL'},
    {'label': 'Belize', 'value': 'BZE'},
    {'label': 'Colombia', 'value': 'COL'},
    {'label': 'Costa Rica', 'value': 'CR'},
    {'label': 'Cuba', 'value': 'CUB'},
    {'label': 'Dominican Republic', 'value': 'DR'},
    {'label': 'Haiti & DR', 'value': 'HAI'},
    {'label': 'Jamaica', 'value': 'JAM'},
    {'label': 'Mexico', 'value': 'MEX'},
    {'label': 'Multiple countries', 'value': 'MULTI'},
    {'label': 'Nicaragua', 'value': 'NIC'},
    {'label': 'Panama', 'value': 'PAN'},
    {'label': 'Peru', 'value': 'PERU'},
    {'label': 'Tanzania', 'value': 'TANZ'},
]
program_choice_options = [
    {'label': 'All programs', 'value': 'ALL'},
    {'label': 'ATR', 'value': 'ATR'},
    {'label': 'Dentistry', 'value': 'DEN'},
    {'label': 'Ecology', 'value': 'ECO'},
    {'label': 'Education', 'value': 'EDU'},
    {'label': 'Gap Year', 'value': 'GAP'},
    {'label': 'Global Health', 'value': 'GH'},
    {'label': 'Hike for Humanity', 'value': 'H4H'},
    {'label': 'Interdisciplinary', 'value': 'ID'},
    {'label': 'Medical', 'value': 'MD'},
    {'label': 'No program', 'value': 'NONE'},
    {'label': 'Nursing', 'value': 'NUR'},
    {'label': 'Optometry', 'value': 'OPT'},
    {'label': 'Pharmacy', 'value': 'PHARM'},
    {'label': 'Physical Therapy', 'value': 'PT'},
    {'label': 'Physician Assistant', 'value': 'PA'},
    {'label': 'Community Enrichment', 'value': 'SERV'},
    {'label': 'Specialized Service', 'value': 'SPCL'},
    {'label': 'Sports without Borders', 'value': 'SWB'},
    {'label': 'Veterinary', 'value': 'VET'},
    {'label': 'WCI', 'value': 'WCI'},
]
year_choice_options = [
    {'label': 'All years', 'value': 'ALL'},
    {'label': '2018', 'value': 2018},
    {'label': '2017', 'value': 2017},
    {'label': '2016', 'value': 2016},
    {'label': '2015', 'value': 2015},
    {'label': '2014', 'value': 2014},
    {'label': '2013', 'value': 2013},
    {'label': '2012', 'value': 2012},
    {'label': '2011', 'value': 2011},
    {'label': '2010', 'value': 2010},
]

# Create map DataFrame.
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
year = 2010
country_choice_single = 'ALL'
country_choice_multi = ['NIC', 'CR', 'MEX']
program_choice_single = 'ALL'
year_choice = 'ALL'
total_volunteers = 0

country_options = ["BZE", "COL", "CR", "DR", "HAI", "IND", "JAM", "MEX", "NIC", "PAN", "PERU", "SPN", "TANZ", "CA", "CUB", "NA", "US", "MULTI"]
fix_country_codes = {'BZE': 'BLZ', 'COL': 'COL', 'CR': 'CRI', 'DR': 'DOM', 'HAI': 'HTI', 'IND': 'IND', 'JAM': 'JAM', 'MEX': 'MEX', 'NIC': 'NIC', 'PAN': 'PAN', 'PERU': 'PER', 'SPN': 'ESP', 'TANZ': 'TZA', 'CA': 'CAN', 'CUB': 'CUB', 'NA': 'NA', 'US': 'USA', 'MULTI': 'MULTI'}
fix_country_names = {'ALL': 'All countries', 'BZE': 'Belize', 'COL': 'Colombia', 'CR': 'Costa Rica', 'DR': 'Dominican Republic', 'HAI': 'Haiti & DR', 'IND': 'India', 'JAM': 'Jamaica', 'MEX': 'Mexico', 'NIC': 'Nicaragua', 'PAN': 'Panama', 'PERU': 'Peru', 'SPN': 'Spain', 'TANZ': 'Tanzania', 'CA': 'Canada', 'CUB': 'Cuba', 'NA': 'NA', 'US': 'USA', 'MULTI': 'Multiple countries'}
fix_program_names = {'ALL': 'All programs', 'NUR': 'Nursing','PT': 'Physical Therapy','GH': 'Global Health','ECO': 'Ecology','MD': 'Medical','ATR': 'ATR','PHARM': 'Pharmacy','WCI': 'Well Child International','SERV': 'Community Enrichment','BIOTECH': 'Biotech','DEN': 'Dentistry','EDU': 'Education','GAP': 'Gap Year','HS': 'High School','H4H': 'Hike for Humanity','INT': 'Internships','OT': 'Occupational Therapy','OPT': 'Optometry','SPCL': 'Specialized Service','PA': 'Physician Assistant','SWB': 'Sports without Borders','VET': 'Veterinary', 'ID': 'Interdisciplinary', 'NONE': 'No program'}
# programs = ['NUR', 'PT', 'GH', 'ECO', 'MD', 'ATR', 'PHARM', 'WCI', 'SERV', 'BIOTECH']
programs = ['ALL', 'NUR', 'PT', 'GH', 'ECO', 'MD', 'ATR', 'PHARM', 'WCI', 'SERV', 'BIOTECH', 'DEN', 'EDU', 'GAP', 'HS', 'H4H', 'INT', 'OT', 'OPT', 'SPCL', 'PA', 'SWB', 'VET', 'ID', 'NONE']
top_programs = [10]
# d = {'Country': None, 'Code': None, 'Number': None}
df = pd.DataFrame()

def Row(class_style, contents):
    return html.Div(className = 'row' + class_style, children = contents)

def Col(size, contents):
    return html.Div(className = size, children = contents)

def load_dataframe():
    yearnums_by_program_df = pd.DataFrame()

    # country = 'CR'

    # Load multiples
    df_by_year_multi = pd.DataFrame()
    for year in years:
        query = "select index, country, volume, program from main_fake where status like %(complete)s and start_date between %(start)s and %(end)s"
        result = read_sql_query(query, conn, params={"complete": '%COMPLETE%', "start": str(year)+"-01-01 00:00:00", "end": str(year)+"-12-31 00:00:00"})
        if not result.empty:
            result.loc[:,'Year'] = pd.Series([year]*result.shape[0], index = result.index)
            # Pull out country repeats.
            df_index = 0
            for entry in result['country']:
                if len(entry) > 5:
                    multiple_countries = entry.split('\\r\\n')
                    if multiple_countries.count(multiple_countries[0]) != len(multiple_countries):
                        result.loc[df_index,'country'] = 'MULTI'
                        if multiple_countries[0] == 'DR':
                            if multiple_countries[1] == 'HAI':
                                result.loc[df_index,'country'] = 'HAI'
                            elif len(multiple_countries) > 2 and multiple_countries[2] == 'HAI':
                                result.loc[df_index,'country'] == 'HAI'
                        df_by_year_multi = df_by_year_multi.append(result.loc[df_index,:], ignore_index = True)
                    elif "\\r\\n" not in result.loc[df_index,'program']:
                        result.loc[df_index,'country'] = multiple_countries[0]
                        df_by_year_multi = df_by_year_multi.append(result.loc[df_index,:], ignore_index = True)
                df_index = df_index + 1

            # Pull out program repeats.
            df_index = 0
            for entry in result['program']:
                multiple_countries = result.loc[df_index,'country'].split('\\r\\n')
                if entry == '':
                    result.loc[df_index,'program'] = 'NONE'
                    df_by_year_multi = df_by_year_multi.append(result.loc[df_index,:], ignore_index = True)
                elif len(entry) > 5:
                    multiple_programs = entry.split('\\r\\n')
                    if multiple_programs.count(multiple_programs[0]) != len(multiple_programs):
                        result.loc[df_index,'program'] = 'ID'
                        result.loc[df_index,'country'] = multiple_countries[0]
                        # print(result)
                        df_by_year_multi = df_by_year_multi.append(result.loc[df_index,:], ignore_index = True)
                        # print("Appending ID")
                    else:

                        result.loc[df_index,'program'] = multiple_programs[0]
                        df_by_year_multi = df_by_year_multi.append(result.loc[df_index,:], ignore_index = True)
                        # print("Appending multi->single")
                        # print(entry)
                        # print(result.loc[df_index,'program'])
                        # print(df_index)
                df_index = df_index + 1

    # Set program to be 'ID' for all multi-program entries.
    # for index in range(df_by_year_multi.shape[0]):
    #     df_by_year_multi.loc[index,'program'] = 'ID'

    # Extract single program entries
    df_by_year = pd.DataFrame()
    for program in programs:
        for year in years:
            # Grab singles
            query = "select country, volume, program from main_fake where status like %(complete)s and start_date between %(start)s and %(end)s and program like %(prog)s"
            result = read_sql_query(query, conn, params={"complete": '%COMPLETE%', "start": str(year)+'-01-01 00:00:00', "end": str(year)+'-12-31 00:00:00', "prog": program})
            # [str(year)+'-01-01', str(year)+'-12-31', program])
            if not result.empty:
                # result['Year'] = pd.Series([year]*result.shape[0], index = result.index)
                result.loc[:,'Year'] = pd.Series([year]*result.shape[0], index = result.index)
                df_by_year = df_by_year.append(result, ignore_index = True)

    # Append multi-program entries to the single program entries
    df_by_year = df_by_year.append(df_by_year_multi, ignore_index = True)

    # print(df_by_year)

    # print(df_by_year_multi.loc[df_by_year_multi['country'] == 'HAI'])
    # print(df_by_year_multi)

    # Clean up DataFrame
    for index, row in df_by_year.iterrows():
        if ("\\r\\n" in row['country']) or ("\\r\\n" in row['program']):
            df_by_year.drop(index, inplace = True)
        elif row['program'] == "":
            df_by_year.drop(index, inplace = True)

    return(df_by_year)

df_full = load_dataframe()

def calculate_total_volunteers():
    total_volunteers = df_full['volume'].astype(int).sum()
    return total_volunteers

total_volunteers = calculate_total_volunteers()

def generate_table(country_choice_single):
    # print(country_choice_single)

    # load_dataframe()

    # query = "select country, program, volume, start_date from main where start_date between '2011-01-01' and '2018-12-31' and program like 'NUR'"
    # query = "select distinct program from main"
    query = "select * from main_fake where country like 'CR'"

    dataframe = read_sql_query(query, conn)
    children = (
        [html.H4(children='Entries: ' + str(len(dataframe)))] +

        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))] +

        # [html.H4('Total number of volunteers: ' + dataframe.iloc[:,3].astype(int).sum().astype(str))] +

        [html.P('Retrieved at: ' + str(datetime.datetime.now()))]
        )
    return children

def generate_map(year):
    df = pd.DataFrame()
    for country in country_options:
        query = "select volume from main_fake where status like %(complete)s and start_date between %(start)s and %(end)s and country like %(country)s"
        result = read_sql_query(query, conn, params={"complete": '%COMPLETE%', "start": str(year)+'-01-01', "end": str(year)+'-12-31', "country": '%'+country+'%'})
        #
        # print(result.iloc[:,0].astype(int).sum())
        # print(result.iloc[:,0])
        d = {'Country': [country], 'Code': [country], 'Number': [result.iloc[:,0].astype(int).sum()]}
        df_temp = pd.DataFrame(data=d)
        df = df.append(df_temp, ignore_index=True)

    df['Code'] = df['Code'].map(fix_country_codes)
    df['Country'] = df['Country'].map(fix_country_names)
    maxcolor_badrbg = colors[round(max(df['Number']))].rgb
    maxcolor_goodrbg = (maxcolor_badrbg[0]*220, maxcolor_badrbg[1]*220, maxcolor_badrbg[2]*220)
    maxcolor = "rgb"+str(maxcolor_goodrbg)
    figure = dict (
        data = [ dict(
                type = 'choropleth',
                locations = df['Code'],
                z = df['Number'],
                text = df['Country'],
                colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
                    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
                # colorscale = [[0, "rgb(220, 220, 220)"], [1, maxcolor]],
                autocolorscale = False,
                reversescale = True,
                marker = dict(
                    line = dict (
                        color = 'rgb(180,180,180)',
                        width = 0.5
                    ) ),
                colorbar = dict(
                    # autotick = False,
                    title = 'Number of Volunteers')
              ) ],

        layout = dict(
            title = ('ISL Volunteers in '+str(year)),
            geo = dict(
                showframe = False,
                showcoastlines = True,
                showcountries = True,
                scope = "world",
                projection = dict(
                    type = 'Mercator'
                ),
                center = dict(
                    lon = -90,
                    lat = 15
                ),
                lonaxis = dict(
                    range = (-120, -25)
                    # showgrid = True
                ),
                lataxis = dict(
                    range = (0, 30)
                    # showgrid = True
                )
            )
        )
    )
    return figure

def generate_vol_num_plot(countries):
    yearnums_by_country_df = pd.DataFrame()
    total_sum = np.zeros(len(years))

    # print(countries)
    for country in countries:
        # print(country)
        df_by_year = pd.DataFrame()
        sum_index = 0
        for year in years:
            result = df_full.loc[df_full['Year'] == year]
            # result = result.loc[result['program'] == program]
            result = result.loc[result['country'] == country]
            # for i in range(result.shape[0]):
            #     if result.loc[i,'program'] == 'NUR':
            # print(result.loc[result['program'] == 'NUR'])

            # query = "select volume from main where status like %(complete)s and start_date between %(start)s and %(end)s and country like %(country)s"
            # result = read_sql_query(query, conn, params={"complete": '%COMPLETE%', "start": str(year)+'-01-01', "end": str(year)+'-12-31', "country": '%'+country+'%'})
            d_by_year = {'Year': [year], 'Number': [result.loc[:,'volume'].astype(int).sum()]}
            df_temp_by_year = pd.DataFrame(data = d_by_year)
            df_by_year = df_by_year.append(df_temp_by_year, ignore_index = True)
            total_sum[sum_index] = total_sum[sum_index] + result.loc[:,'volume'].astype(int).sum()
            sum_index = sum_index + 1
        # print('Printing data field by year: ')
        # print(df_by_year)
        d_by_country = {'Country': [country], 'DataFrame': [df_by_year]}
        df_temp_by_country = pd.DataFrame(data = d_by_country)
        yearnums_by_country_df = yearnums_by_country_df.append(df_temp_by_country, ignore_index = True)
        # print('Complicated version:')
        # print(yearnums_by_country_df)

    color_counter = 0
    colors = {
        'Lines': ['rgba(15,32,88,1)', 'rgba(5,77,99,1)', 'rgba(26,154,139,1)', 'rgba(176,95,161,1)', 'rgba(181,203,153,1)', 'rgba(255,210,218,1)', 'rgba(34,34,59,1)', 'rgba(154,140,152,1)', 'rgba(201,173,167,1)'],
        'Bars': ['rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)']}
    figure_data = []
    for country in countries:
        df = yearnums_by_country_df.loc[yearnums_by_country_df['Country'] == country].iloc[0,1]

        hoverNumber = df['Number'].astype('str').values

        temp_data_1 = dict(
            type = 'scatter',
            mode = 'line',
            name = fix_country_names[country],
            x = df['Year'],
            y = df['Number'],
            line = dict(shape = 'spline', color = colors['Lines'][color_counter]),
            hoverinfo = 'text',
            hovertext = ''
            )
        temp_data_2 = dict(
            type = 'bar',
            x = df['Year'],
            y = df['Number'],
            name = fix_country_names[country],
            marker = dict(color = colors['Bars'][color_counter]),
            showlegend = False,
            hovertext = fix_country_names[country] + " - " + hoverNumber,
            # hovertext = fix_country_names[country],
            hoverinfo = 'text'
            )
        figure_data.append(temp_data_1)
        figure_data.append(temp_data_2)
        color_counter = color_counter + 1

    temp_data_3 = dict(
        type = 'scatter',
        mode = 'text',
        name = '',
        x = years,
        y = total_sum,
        # line = dict(shape = 'spline', color = colors['Lines'][color_counter]),
        hovertext = '',
        hoverinfo = 'none',
        autorange = False,
        # connectgaps = True
        text = total_sum,
        textposition = 'top center',
        textfont = dict(
            size = 18,
        ),
        cliponaxis = False
    )
    figure_data.append(temp_data_3)

    figure = dict(
        data = figure_data,
        layout = dict(
            xaxis = dict(autotick = False, showgrid = False, showline = False, fixedrange = True),
            yaxis = dict(autorange = True, type = 'linear', showgrid = False, showline = False, ticks = '', showticklabels = False, fixedrange = True),
            # barmode = '',
            showlegend = False,
            margin = dict(t = 30, l = 10, r = 10),
            barmode = 'stack',
        )
    )
    return figure

def generate_country_breakdown(country):
    color_counter = 0
    colors = {
        'Lines': ['rgba(15,32,88,1)', 'rgba(5,77,99,1)', 'rgba(26,154,139,1)', 'rgba(176,95,161,1)', 'rgba(181,203,153,1)', 'rgba(255,210,218,1)', 'rgba(34,34,59,1)', 'rgba(154,140,152,1)', 'rgba(201,173,167,1)', 'rgba(15,32,88,1)', 'rgba(5,77,99,1)', 'rgba(26,154,139,1)', 'rgba(176,95,161,1)', 'rgba(181,203,153,1)', 'rgba(255,210,218,1)', 'rgba(34,34,59,1)', 'rgba(154,140,152,1)', 'rgba(201,173,167,1)', 'rgba(15,32,88,1)', 'rgba(5,77,99,1)', 'rgba(26,154,139,1)', 'rgba(176,95,161,1)', 'rgba(181,203,153,1)', 'rgba(255,210,218,1)', 'rgba(34,34,59,1)', 'rgba(154,140,152,1)', 'rgba(201,173,167,1)'],
        'Bars': ['rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)', 'rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)', 'rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)']}
    figure_data = []
    program_by_year = pd.DataFrame()

    total_sum = np.zeros(len(years))
    # total_sum = {'year': years, 'total': np.zeros(len(programs))}
    # total_sum = pd.DataFrame(data = total_sum)
    # total_sum.loc[:,'volume'] = pd.Series(0*len(programs))
    # yearnums_by_program_df = pd.DataFrame(data = {'Program': [], 'Country': [], 'Year': [], 'Number': []})

    if country == "ALL":
        country_result = df_full
        # global total_volunteers
        # total_volunteers = df_full['volume'].astype(int).sum()
    else:
        country_result = df_full.loc[df_full['country'] == country]

    for program in programs:
        program_result = country_result.loc[country_result['program'] == program]
        df_by_year = pd.DataFrame()
        sum_index = 0
        for year in years:
            result = program_result.loc[program_result['Year'] == year]

            # if program == 'NUR':
            #     print('Program:' + program)
            #     print(result)
            if result.loc[:,'volume'].astype(int).sum() > 0:
                d_by_year = {'Year': [year], 'Number': [result.loc[:,'volume'].astype(int).sum()]}
                df_temp_by_year = pd.DataFrame(data = d_by_year)
                df_by_year = df_by_year.append(df_temp_by_year, ignore_index = True)
                # total_sum.loc[total_sum['year'] == year] = total_sum.loc[total_sum['year'] == year] + df_by_year.loc[:,'Number']
                total_sum[sum_index] = total_sum[sum_index] + result.loc[:,'volume'].astype(int).sum()
            sum_index = sum_index + 1

        if not df_by_year.empty:
            # total_sum.loc[total_sum['program'] == program] = total_sum.loc[total_sum['program'] == program] + df_by_year.loc[:,'Number']
            # total_sum[]
            # total_sum[df_index] = total_sum[df_index] + df_by_year['Number'].astype('int')
            hoverNumber = df_by_year['Number'].astype('str').values
            temp_data_1 = dict(
                type = 'scatter',
                name = fix_program_names[program],
                x = df_by_year['Year'],
                y = df_by_year['Number'],
                line = dict(shape = 'spline', color = colors['Lines'][color_counter]),
                hovertext = '',
                hoverinfo = 'text',
                connectgaps = True
                )
            temp_data_2 = dict(
                type = 'bar',
                x = df_by_year['Year'],
                y = df_by_year['Number'],
                name = fix_program_names[program],
                marker = dict(color = colors['Bars'][color_counter]),
                showlegend = False,
                hovertext = fix_program_names[program] + " - " + hoverNumber,
                hoverinfo = 'text',
                autorange = False,
                # text = total_sum,
                textposition = 'auto'
                )
            figure_data.append(temp_data_1)
            figure_data.append(temp_data_2)
            color_counter = color_counter + 1

    temp_data_3 = dict(
        type = 'scatter',
        mode = 'text',
        name = '',
        x = years,
        y = total_sum,
        # line = dict(shape = 'spline', color = colors['Lines'][color_counter]),
        hovertext = '',
        hoverinfo = 'none',
        autorange = False,
        # connectgaps = True
        text = total_sum,
        textposition = 'top center',
        textfont = dict(
            size = 18,
        ),
        cliponaxis = False
    )
    figure_data.append(temp_data_3)

    figure = dict(
        data = figure_data,
        layout = dict(
            # title = sum(total_sum),
            xaxis = dict(autotick = False, showgrid = False, showline = False, fixedrange = True),
            yaxis = dict(autorange = True, type = 'linear', showgrid = False, showline = False, ticks = '', showticklabels = False, fixedrange = True),
            barmode = 'stack',
            showlegend = False,
            margin = dict(t = 30, l = 10, r = 10),
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)'
        )
    )
    return figure

def generate_program_breakdown(program):
    # print('Program:' + program)
    yearnums_by_country_df = pd.DataFrame()
    total_sum = np.zeros(len(years))
    if program == 'ALL':
        program_result = df_full
    else:
        program_result = df_full.loc[df_full['program'] == program]
    for country in country_options:
        sum_index = 0
        country_result = program_result.loc[program_result['country'] == country]
        df_by_year = pd.DataFrame()
        for year in years:
            result = country_result.loc[country_result['Year'] == year]
            total_sum[sum_index] = total_sum[sum_index] + result.loc[:,'volume'].astype(int).sum()
            sum_index = sum_index + 1
            if result.loc[:,'volume'].astype(int).sum() > 0:
                d_by_year = {'Year': [year], 'Number': [result.loc[:,'volume'].astype(int).sum()]}
            else:
                d_by_year = {'Year': [year], 'Number': ['NaN']}
            df_temp_by_year = pd.DataFrame(data = d_by_year)
            df_by_year = df_by_year.append(df_temp_by_year, ignore_index = True)
        d_by_country = {'Country': [country], 'DataFrame': [df_by_year]}
        df_temp_by_country = pd.DataFrame(data = d_by_country)
        yearnums_by_country_df = yearnums_by_country_df.append(df_temp_by_country, ignore_index = True)

    color_counter = 0
    colors = {
        'Lines': ['rgba(15,32,88,1)', 'rgba(5,77,99,1)', 'rgba(26,154,139,1)', 'rgba(176,95,161,1)', 'rgba(181,203,153,1)', 'rgba(255,210,218,1)', 'rgba(34,34,59,1)', 'rgba(154,140,152,1)', 'rgba(201,173,167,1)', 'rgba(15,32,88,1)', 'rgba(5,77,99,1)', 'rgba(26,154,139,1)', 'rgba(176,95,161,1)', 'rgba(181,203,153,1)', 'rgba(255,210,218,1)', 'rgba(34,34,59,1)', 'rgba(154,140,152,1)', 'rgba(201,173,167,1)', 'rgba(15,32,88,1)', 'rgba(5,77,99,1)', 'rgba(26,154,139,1)', 'rgba(176,95,161,1)', 'rgba(181,203,153,1)', 'rgba(255,210,218,1)', 'rgba(34,34,59,1)', 'rgba(154,140,152,1)', 'rgba(201,173,167,1)'],
        'Bars': ['rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)', 'rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)', 'rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)']}
    figure_data = []
    for country in country_options:
        df = yearnums_by_country_df.loc[yearnums_by_country_df['Country'] == country].iloc[0,1]
        hoverNumber = df['Number'].astype('str').values
        pleasePlot = False
        for number in hoverNumber:
            if number != 'NaN':
                pleasePlot = True

        if pleasePlot == True:

            temp_data_1 = dict(
                type = 'scatter',
                name = fix_country_names[country],
                x = df['Year'],
                y = df['Number'],
                line = dict(shape = 'spline', color = colors['Lines'][color_counter]),
                hovertext = '',
                hoverinfo = 'text',
                connectgaps = True
                )
            temp_data_2 = dict(
                type = 'bar',
                x = df['Year'],
                y = df['Number'],
                name = fix_country_names[country],
                marker = dict(color = colors['Bars'][color_counter]),
                showlegend = False,
                hovertext = fix_country_names[country] + " - " + hoverNumber,
                hoverinfo = 'text'
                )
            figure_data.append(temp_data_1)
            figure_data.append(temp_data_2)
            color_counter = color_counter + 1

    temp_data_3 = dict(
        type = 'scatter',
        mode = 'text',
        name = '',
        x = years,
        y = total_sum,
        # line = dict(shape = 'spline', color = colors['Lines'][color_counter]),
        hovertext = '',
        hoverinfo = 'none',
        autorange = False,
        # connectgaps = True
        text = total_sum,
        textposition = 'top center',
        textfont = dict(
            size = 18,
        ),
        cliponaxis = False
    )
    figure_data.append(temp_data_3)

    figure = dict(
        data = figure_data,
        layout = dict(
            title = '',
            xaxis = dict(autotick = False, showgrid = False, showline = False, fixedrange = True),
            yaxis = dict(autorange = True, type = 'linear', showgrid = False, showline = False, ticks = '', showticklabels = False, fixedrange = True),
            barmode = 'stack',
            showlegend = False,
            margin = dict(t = 30, l = 10, r = 10),
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)'
        )
    )
    return figure

def generate_country_percent_of_total(country, year):
    local_df = df_full[df_full['country'] == country]
    if year != 'ALL':
        local_df = local_df[local_df['Year'] == year]
    country_total = local_df['volume'].astype(int).sum()
    country_percent = int(round(country_total/total_volunteers*100))
    figure_data = [dict(
        type = 'pie',
        values = [country_total,total_volunteers - country_total],
        labels = [fix_country_names[country],'Other countries'],
        marker = dict(colors = ['rgba(15,32,88,0.7)', 'rgba(185,184,190,1)']),
        # hoverinfo = 'value',
        textinfo = 'none',
        # textfont = dict(
        #     size = 18,
        # ),
        hole = 0.4,
        # cliponaxis = False
    )]

    if country_percent < 1:
        percent_text = "<1%"
    else:
        percent_text = str(country_percent) + "%"

    figure = dict(
        data = figure_data,
        layout = dict(
            showlegend = False,
            margin = dict(t = 25, l = 0, r = 0, b = 25, pad = 0),
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
            height = 200,
            annotations = [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": percent_text,
                "x": 0.5,
                "y": 0.5
            },
            ]
        )
    )
    return figure

def generate_country_top_programs(country, year):
    local_df = df_full[df_full['country'] == country]
    # global top_programs
    top_programs_numbers = []
    top_programs_labels = []
    for program in programs:
        program_df = local_df[local_df['program'] == program]
        # print("Program: " + str(program))
        # print(program_df)
        top_programs_numbers.append(program_df['volume'].astype(int).sum())
        top_programs_labels.append(fix_program_names[program])
    # print('Programs: ')
    # print(top_programs)
    figure_data = [dict(
        type = 'pie',
        values = top_programs_numbers,
        labels = top_programs_labels,
        marker = dict(colors = ['rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)', 'rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)', 'rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)']),
        hoverinfo = 'labels',
        # text = top_programs_labels,
        textinfo = 'none',
        # textfont = dict(
        #     size = 18,
        # ),
        hole = 0.4,
        cliponaxis = False,
    )]

    figure = dict(
        data = figure_data,
        layout = dict(
            showlegend = False,
            margin = dict(t = 25, l = 0, r = 0, b = 25, pad = 0),
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
            height = 200,
            # annotations = [
            # {
            #     "font": {
            #         "size": 20
            #     },
            #     "showarrow": False,
            #     "text": fix_country_names[country],
            #     "x": 0.5,
            #     "y": 0.5
            # },
            # ]
        )
    )
    return figure

def generate_program_percent_of_total(program, year):
    local_df = df_full[df_full['program'] == program]
    if year != 'ALL':
        local_df = local_df[local_df['Year'] == year]
    program_total = local_df['volume'].astype(int).sum()
    program_percent = int(round(program_total/total_volunteers*100))
    figure_data = [dict(
        type = 'pie',
        values = [program_total,total_volunteers - program_total],
        labels = [fix_program_names[program],'Other countries'],
        marker = dict(colors = ['rgba(15,32,88,0.7)', 'rgba(185,184,190,1)']),
        # hoverinfo = 'value',
        textinfo = 'none',
        # textfont = dict(
        #     size = 18,
        # ),
        hole = 0.4,
        # cliponaxis = False
    )]

    if program_percent < 1:
        percent_text = "<1%"
    else:
        percent_text = str(program_percent) + "%"

    figure = dict(
        data = figure_data,
        layout = dict(
            showlegend = False,
            margin = dict(t = 25, l = 0, r = 0, b = 25, pad = 0),
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
            height = 200,
            annotations = [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": percent_text,
                "x": 0.5,
                "y": 0.5
            },
            ]
        )
    )
    return figure

def generate_program_top_countries(program, year):
    local_df = df_full[df_full['program'] == program]
    # global top_programs
    top_countries_numbers = []
    top_countries_labels = []
    for country in country_options:
        country_df = local_df[local_df['country'] == country]
        # print("Program: " + str(program))
        # print(program_df)
        top_countries_numbers.append(country_df['volume'].astype(int).sum())
        top_countries_labels.append(fix_country_names[country])
    # print('Programs: ')
    # print(top_programs)
    figure_data = [dict(
        type = 'pie',
        values = top_countries_numbers,
        labels = top_countries_labels,
        marker = dict(colors = ['rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)', 'rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)', 'rgba(15,32,88,0.7)', 'rgba(5,77,99,0.7)', 'rgba(26,154,139,0.7)', 'rgba(176,95,161,0.7)', 'rgba(181,203,153,0.7)', 'rgba(255,210,218,0.7)', 'rgba	(34,34,59,0.7)', 'rgba(154,140,152,0.7)', 'rgba(201,173,167,0.7)']),
        hoverinfo = 'labels',
        # text = top_programs_labels,
        textinfo = 'none',
        # textfont = dict(
        #     size = 18,
        # ),
        hole = 0.4,
        cliponaxis = False,
    )]

    figure = dict(
        data = figure_data,
        layout = dict(
            showlegend = False,
            margin = dict(t = 25, l = 0, r = 0, b = 25, pad = 0),
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
            height = 200,
            # annotations = [
            # {
            #     "font": {
            #         "size": 20
            #     },
            #     "showarrow": False,
            #     "text": fix_country_names[country],
            #     "x": 0.5,
            #     "y": 0.5
            # },
            # ]
        )
    )
    return figure

def country_choice_dropdown(tab_id, multi_choice=False, choose_countries=True, dropdown_options=country_choice_options):
    if multi_choice == True and choose_countries == True:
        value = ['NIC', 'CR', 'MEX']
    elif multi_choice == False and choose_countries == True:
        value = 'ALL'
    elif multi_choice == False and choose_countries == False:
        value = 'ALL'
    return dcc.Dropdown(
        id = tab_id,
        options = dropdown_options,
        value = value,
        multi = multi_choice
    )

def year_choice_dropdown(dropdown_id, dropdown_options = year_choice_options):
    return dcc.Dropdown(
        id = dropdown_id,
        options = dropdown_options,
        value = 'ALL',
        multi = False
    )

app = dash.Dash('app', meta_tags = [{'name': 'viewport', 'content': 'width=device-width, initial-scale=1, maximum-scale=1'}])
# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )
app.config['suppress_callback_exceptions'] = True
app.title = 'ISL Dashboard'
# app.meta_tags = [{"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1"}]
# app.css.append_css({'external_url': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css'})
server = app.server

def serve_layout_additional_country_details(country_choice_single, emtpy_card):
    if emtpy_card == True:
        print('Empty card received.')
        returned_card = [html.Div(className = 'card shadow-sm', children = [
                    html.Div(className = 'card-header', children = 'Additional country details'),
                    html.Div(className = 'card-body', children = [

                        html.Br(),
                        Row(' align-items-end',[
                            Col('col-sm-12',html.P(className = 'text-center', children = 'Please select a country to view additional details.'))

                        ])
                    ])
                ])
                ]
    # if country_choice_single == 'ALL':
    #     returned_card = html.Div()
    else:
        returned_card = [html.Div(className = 'card shadow-sm', children = [
                    html.Div(className = 'card-header', children = 'Additional country details'),
                    html.Div(className = 'card-body', children = [

                        html.Br(),
                        Row(' align-items-end',[
                            Col('col-sm-6', [
                                Row('',[
                                    Col('col-sm-12',html.P(className = 'text-center', children = 'Percentage of total volunteers who volunteer in ' +  fix_country_names[country_choice_single]))
                                ]),
                                Row('',[
                                    Col('col-sm-12',html.Div(dcc.Graph(
                                        id = 'graph_country_percent_of_total',
                                        figure = generate_country_percent_of_total(country_choice_single, year_choice),
                                        config = config
                                        ),
                                        # style={'min-width': '300px'}
                                        )),
                                ]),
                            ]),
                            Col('col-sm-6', [
                                Row('',[
                                    Col('col-sm-12',[html.P(className = 'text-center', children = 'Programs in ' +  fix_country_names[country_choice_single]), html.Br()])
                                ]),
                                Row('',[
                                    Col('col-sm-12',html.Div(dcc.Graph(
                                        id = 'graph_country_top_programs',
                                        # figure = generate_country_percent_of_total(country_choice_single),
                                        figure = generate_country_top_programs(country_choice_single, year_choice),
                                        config = config
                                        ),
                                        # style={'min-width': '300px'}
                                        )),
                                ]),
                            ]),
                        ])
                    ])
                ])
                ]

    return returned_card

def serve_layout_additional_program_details(program_choice_single):
    if program_choice_single == 'ALL':
        returned_card = html.Div()
    else:
        returned_card = [html.Div(className = 'card shadow-sm', children = [
                    html.Div(className = 'card-header', children = 'Additional program details'),
                    html.Div(className = 'card-body', children = [

                        html.Br(),
                        Row(' align-items-end',[
                            Col('col-sm-6', [
                                Row('',[
                                    Col('col-sm-12',html.P(className = 'text-center', children = 'Percentage of total volunteers who volunteer in ' +  fix_program_names[program_choice_single]))
                                ]),
                                Row('',[
                                    Col('col-sm-12',html.Div(dcc.Graph(
                                        id = 'graph_program_percent_of_total',
                                        figure = generate_program_percent_of_total(program_choice_single, year_choice),
                                        config = config
                                        ),
                                        # style={'min-width': '300px'}
                                        )),
                                ]),
                            ]),
                            Col('col-sm-6', [
                                Row('',[
                                    Col('col-sm-12',[html.P(className = 'text-center', children = 'Countries with volunteers in ' +  fix_program_names[program_choice_single])])
                                ]),
                                Row('',[
                                    Col('col-sm-12',html.Div(dcc.Graph(
                                        id = 'graph_program_top_programs',
                                        # figure = generate_country_percent_of_total(country_choice_single),
                                        figure = generate_program_top_countries(program_choice_single, year_choice),
                                        config = config
                                        ),
                                        # style={'min-width': '300px'}
                                        )),
                                ]),
                            ]),
                        ])
                    ])
                ])
                ]

        return returned_card

def serve_layout_main(country_choice_single, program_choice_single):
    return html.Div(id = 'main_page',
        children=[
        html.Nav(className = 'navbar navbar-expand-md navbar-light bg-light border-bottom shadow-sm align-items-center', children = [
            html.A(className = 'navbar-brand', href = '#', children = [html.A(className = 'navbar-brand d-inline pl-3 pt-3', children = 'Dashboard')])
        ]),
        html.Div(className = 'container', children = [
            # Overall numbers
            Row('',[
                Col('col-12 mt-4', [
                    # html.Div(className = 'shadow-sm rounded border p-1 bg-light', children = [
                    html.Div(className = 'card shadow-sm', children = [
                        html.Div(className = 'card-header', children = 'Total volunteer numbers by country'),
                        html.Div(className = 'card-body', children = [

                            # Row('',[
                            #     Col('col-md-12',html.Div(className = 'text-center mt-4', children = html.H1(className = 'display-5', children = 'Total volunteer numbers by country'))),
                            # ]),
                            Row(' mt-3',[
                                Col('col-sm col-md',''),
                                Col('col-sm-8 col-md-8',html.H4(children = html.Small(className = 'text-muted', children = 'Select countries:'))),
                                Col('col-sm col-md','')
                            ]),
                            Row('',[
                                Col('col-sm col-md',''),
                                Col('col-sm-8 col-md-8',country_choice_dropdown('vol_num_plot_country_dropdown', multi_choice=True)),
                                Col('col-sm col-md','')
                            ]),

                            Row('',[
                                Col('col-sm col-md',''),
                                Col('col-sm-10 col-md-12',html.Div(dcc.Graph(
                                    id = 'vol_num_plot',
                                    # figure = generate_country_breakdown(country_choice_single),
                                    config = config,
                                    # style = {'height': '80vh'}
                                    ),style={'min-width': '300px'})),
                                Col('col-sm col-md','')
                            ])
                        ])
                    ])
                ])
            ]),
            # Detail rows
            Row(' mb-4 mt-4',[
            # Country details
                Col('col-sm-12 col-md-6 p-3',[
                    html.Div(className = 'card shadow-sm', children = [
                    # html.Div(className = 'shadow-sm rounded border p-1 bg-light', children = [
                        html.Div(className = 'card-header', children = 'Breakdown of country by programs'),
                        html.Div(className = 'card-body', children = [
                            Row(' mt-3',[
                                Col('col-sm col-md',''),
                                Col('col-sm-8 col-md-8',html.H4(children = html.Small(className = 'text-muted', children = 'Select a country:'))),
                                Col('col-sm col-md','')
                            ]),
                            Row('',[
                                Col('col-sm col-md',''),
                                Col('col-sm-8 col-md-8',country_choice_dropdown('country_breakdown_dropdown', multi_choice=False)),
                                Col('col-sm col-md','')
                            ]),
                            Row('',[
                                Col('col-md-12',html.Div(className = 'text-center mt-4', children = html.H1(id = 'country_breakdown_title', className = 'display-5', children = fix_country_names[country_choice_single]))),
                            ]),
                            Row('',[
                                Col('col-sm col-md',''),
                                Col('col-sm-10 col-md-12',html.Div(dcc.Graph(
                                    id = 'country_breakdown_graph',
                                    # figure = generate_country_breakdown(country_choice_single),
                                    config = config,
                                    # style = {'height': '80vh'}
                                    ),style={'min-width': '300px'})),
                                Col('col-sm col-md','')
                            ])
                        ])
                    ])
                ]),
                # Additional country details
                html.Div(id = 'additional_country_details'),
            # Program details
                Col('col-sm-12 col-md-6 p-3',[
                    html.Div(className = 'card shadow-sm', children = [
                        html.Div(className = 'card-header', children = 'Breakdown of program by countries'),
                        html.Div(className = 'card-body', children = [
                            Row(' mt-3',[
                                Col('col-sm col-md',''),
                                Col('col-sm-8 col-md-8',html.H4(children = html.Small(className = 'text-muted', children = 'Select a program:'))),
                                Col('col-sm col-md','')
                            ]),
                            Row('',[
                                Col('col-sm col-md',''),
                                Col('col-sm-8 col-md-8',country_choice_dropdown('program_breakdown_dropdown', multi_choice=False, choose_countries=False, dropdown_options=program_choice_options)),
                                Col('col-sm col-md','')
                            ]),
                            Row('',[
                                Col('col-md-12',html.Div(className = 'text-center mt-4', children = html.H1(id = 'program_breakdown_title', className = 'display-5', children = fix_program_names[program_choice_single]))),
                            ]),
                            Row('',[
                                Col('col-sm col-md',''),
                                Col('col-sm-10 col-md-12',html.Div(dcc.Graph(
                                    id = 'program_breakdown_graph',
                                    figure = generate_program_breakdown(program_choice_single),
                                    config = config
                                    ),
                                    style={'min-width': '300px'}
                                    )),
                                Col('col-sm col-md','')
                            ])
                        ])
                    ])
                ]),
                # Additional program details
                html.Div(id = 'additional_program_details'),
            ]),
            # html.Div(className = 'row', id = 'additional_country_details')

        ]),
    ])

app.layout = serve_layout_main(country_choice_single, program_choice_single)

@app.callback(
    Output('table', 'children'),
    [Input('team_data_country_dropdown', 'value')]
)
def update_table_country(country_choice_single):
    return generate_table(country_choice_single)

@app.callback(
    Output('vol_num_plot', 'figure'),
    [Input('vol_num_plot_country_dropdown', 'value')]
)
def update_vol_num_plot(country_choice_multi):
    return generate_vol_num_plot(country_choice_multi)

# Update country details page.
@app.callback(
    Output('country_breakdown_title', 'children'),
    [Input('country_breakdown_dropdown', 'value')]
)
def update_country_breakdown_title(country_choice_single):
    return fix_country_names[country_choice_single]

@app.callback(
    Output('country_breakdown_graph', 'figure'),
    [Input('country_breakdown_dropdown', 'value')]
)
def update_country_breakdown_graph(country_choice_single):
    return generate_country_breakdown(country_choice_single)

# Update additional country details card.
@app.callback(
    Output('additional_country_details', 'children'),
    [Input('country_breakdown_dropdown', 'value'), Input('program_breakdown_dropdown', 'value')]
)
def update_additional_country_details(country_choice_single, program_choice):
    if country_choice_single == 'ALL' and program_choice != 'ALL':
        empty_card = True
        print('Emtpy card')
    elif country_choice_single == 'ALL' and program_choice == 'ALL':
        return None
    else:
        empty_card = False
    return serve_layout_additional_country_details(country_choice_single, empty_card)

@app.callback(
    Output('additional_country_details', 'className'),
    [Input('country_breakdown_dropdown', 'value'), Input('program_breakdown_dropdown', 'value')]
)
def update_additional_country_details(country_choice_single, program_choice):
    print(country_choice_single)
    print(program_choice)
    # if country_choice_single == 'ALL':
    #     className = ''
    if program_choice != 'ALL' or country_choice_single != 'ALL':
        className = 'col-md-6 p-3'
    else:
        className = ''

    print(className)
    return className

# @app.callback(
#     Output('graph_country_percent_of_total', 'figure'),
#     [Input('country_breakdown_dropdown', 'value')]
# )
# def update_country_percent_of_total(country_choice_single):
#     return serve_layout_additional_country_details(country_choice_single)




# Update program details page.
@app.callback(
    Output('program_breakdown_title', 'children'),
    [Input('program_breakdown_dropdown', 'value')]
)
def update_program_breakdown_title(program_choice_single):
    return fix_program_names[program_choice_single]

@app.callback(
    Output('program_breakdown_graph', 'figure'),
    [Input('program_breakdown_dropdown', 'value')]
)
def update_program_breakdown_graph(program_choice_single):
    return generate_program_breakdown(program_choice_single)

# Update additional program details card.
@app.callback(
    Output('additional_program_details', 'children'),
    [Input('program_breakdown_dropdown', 'value')]
)
def update_additional_program_details(program_choice_single):
    return serve_layout_additional_program_details(program_choice_single)

@app.callback(
    Output('additional_program_details', 'className'),
    [Input('program_breakdown_dropdown', 'value')]
)
def update_additional_program_details(program_choice_single):
    if program_choice_single == 'ALL':
        className = ''
    else:
        className = 'col-md-6 p-3'
    return className



@app.callback(
		Output('choropleth', 'figure'),
		[Input('year-slider', 'value')])
def update_map(year):
    return generate_map(year)

if __name__ == '__main__':
    app.run_server(debug=True)
