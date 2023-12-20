import polars as pl
import altair as alt
import streamlit as st

import altair_transform
from kneed import KneeLocator
import pandas as pd

st.set_page_config('Curva Tarefas', layout='wide')

@st.cache_resource
def load_data():
    #schema = {'cod_unb': pl.Int64, 'cod_pdv': pl.Int64, 'id_sku': pl.Int64, 'segmentacao_primaria': pl.Utf8, 'task_text': pl.Utf8, 'visit_date': pl.Utf8, 'task_coins': pl.Int64, 'QTDE TASKS TOTAIS': pl.Int64, '% TASKS VALIDADAS': pl.Utf8, '% Pontos Validados': pl.Utf8} 
    #data = pl.scan_csv('pontos_bees/*.csv', separator=';', schema=schema)
    #data = data.with_columns([pl.col(x).str.replace(',','.').cast(pl.Float32) for x in ['% TASKS VALIDADAS','% Pontos Validados']])
    #data.collect().write_parquet('pontos_bees/pontos.parquet')
    data = pl.scan_parquet('pontos_bees/*.parquet')
    cliente = pl.scan_parquet('data/cliente.parquet')
    seg_consolidado = pl.scan_parquet('data/seg_consolidado.parquet')
    produtos = pl.scan_parquet('data/produtos.parquet')
    cestas = pl.scan_parquet('data/cestas.parquet')
    
    return data, cliente, seg_consolidado, produtos, cestas

data, cliente, seg_consolidado, produtos, cestas = load_data()

lista_cestas = cestas.select(pl.col('Cesta').unique().sort()).collect().to_series().to_list()
lista_segmento = ['Todos'] + data.select(pl.col('segmentacao_primaria').unique().sort()).collect().to_series().to_list()

col1, col2, col3 = st.columns(3)
with col1:
    st.selectbox('Cestas', lista_cestas, index=lista_cestas.index('SPATEN'), key='cesta')
with col2:
    st.selectbox('Segmento', lista_segmento, index=lista_segmento.index('BAR'), key='seg')
with col3:
    col4, col5 = st.columns(2)
    with col4:
        st.number_input('Caixas Maior Igual:', 1, 999, 1, step=1, key='min')
    with col5:
        st.number_input('Caixas Menor:', 1, 999, 999, step=1, key='max')

if st.session_state['seg'] != 'Todos':
    data = data.filter(pl.col('segmentacao_primaria') == st.session_state['seg'])

if st.session_state['cesta']:
    lista_skus = cestas.filter(pl.col('Cesta') == st.session_state['cesta']).select(pl.col('cod_sku').unique().sort()).collect().to_series().to_list()
    data = data.filter(pl.col('id_sku').is_in(lista_skus))
    data = data.with_columns([
        pl.when(pl.col('task_text').str.contains('de \d+ caixa'))\
        .then(pl.col('task_text').str.extract('de (\d+) caixa',1))\
        .otherwise(1).cast(pl.Int16).alias('cxs')
    ]).with_columns(
        pl.col('task_coins').qcut([x/10 for x in range(1,11)]).cast(pl.Utf8).str.extract_all('(\d+)').list.join('-').alias('bins')
    ).with_columns(
        pl.when(~pl.col('bins').str.contains('-'))\
        .then(pl.concat_str([pl.lit('10-'), pl.col('bins')]))\
        .otherwise(pl.col('bins')).alias('bins')
    ).filter(
        (pl.col('cxs') >= st.session_state['min']) &
        (pl.col('cxs') < st.session_state['max'])
    )
    
    data = data.group_by(['bins', 'task_coins']).agg([
        pl.col('QTDE TASKS TOTAIS').sum().alias('quantidade'),
        pl.col('% TASKS VALIDADAS').sum().alias('validadas'),
        (pl.col('% TASKS VALIDADAS').sum()/pl.col('QTDE TASKS TOTAIS').sum()).alias('conversão')
    ]).with_columns(
        pl.col('bins').str.split('-').list[0].cast(pl.Int16).alias('sort_number')
    ).sort('sort_number')


final = data\
    .filter(pl.col('conversão').is_between(0.05, 0.9))\
    .filter(pl.col('task_coins').is_between(50, 500))\
    .collect()\
    .to_pandas()

chart = alt.Chart(final).mark_point().encode(
    x = alt.X('task_coins', sort=alt.SortField('sort_number')),
    y = alt.Y('conversão').scale(zero=True),
).properties(width = 1200, height = 600)

line = chart.transform_regression('task_coins','conversão', order = 3, method='poly', extent=[50,500], params = False).mark_line()
params = chart.transform_regression('task_coins','conversão', order = 3, method='poly', extent=[50,500], params = True).mark_line()


coefs = altair_transform.extract_data(params)

points_x = list(range(50,500))
points_y = [round(coefs['coef'][0][3]*x**3 + coefs['coef'][0][2]*x**2 + coefs['coef'][0][1]*x + coefs['coef'][0][0],3) for x in points_x]

if coefs['coef'][0][3] > 0:
    c = 'concave'
else:
    c = 'convex'

kl = KneeLocator(points_x, points_y, curve=c)
#st.write(kl.knee_y, points_x[points_y.index(round(kl.knee_y,3))])

point = alt.Chart(final).mark_point(color = 'red', shape = 'cross').encode(
    x = alt.datum(points_x[points_y.index(kl.knee_y)]),
    y = alt.datum(kl.knee_y)
)

col1, col2 = st.columns([3,1])

with col1:
    st.altair_chart((chart + line + point))
with col2:
    st.metric('Quantidade de Tarefas:', final['quantidade'].sum())
    st.metric('Quantidade Ideal de Pontos:', points_x[points_y.index(round(kl.knee_y,3))])
    
st.dataframe(final)

    




