"""
Structured data extracted from PDF and Helgi files in the workspace.
This module provides real Sri Lankan paddy production, weather, and disease data
sourced from official government PDFs:
  - Yala2024Metric.pdf (2024 Yala season production by district)
  - 2024_2025Maha_Metric.pdf (2024/2025 Maha season production by district)
  - 1.3.pdf (Annual & monthly average air temperature 2018-2023)
  - 1.5.pdf (Average relative humidity 2019-2023)
  - 1.6.pdf (Annual & monthly rainfall 2018-2023)
  - RRDI disease PDFs (8 rice diseases)
  - chart.helgi / chart(1).helgi (Helgi Library metadata)
"""

# ============================================================================
# HELGI LIBRARY CHART METADATA
# These .helgi files are JSON chart configuration for Helgi Library.
# They do NOT contain actual data values, only references to indicators.
# ============================================================================
HELGI_METADATA = {
    'rice_consumption': {
        'source_file': 'chart.helgi',
        'title': 'Rice Consumption (Total) in Sri Lanka',
        'period': '1961 - 2023',
        'unit': 'kt (kilotonnes)',
        'indicator_id': 864,
        'country_id': 154,  # Sri Lanka
    },
    'rice_production': {
        'source_file': 'chart (1).helgi',
        'title': 'Rice Production in Sri Lanka',
        'period': '1961 - 2024',
        'unit': 'kt (kilotonnes)',
        'indicator_id': 1244,
        'country_id': 154,  # Sri Lanka
    }
}

# ============================================================================
# DISTRICT-LEVEL YIELD DATA FROM PRODUCTION PDFs (Metric Units)
# Source: Yala2024Metric.pdf & 2024_2025Maha_Metric.pdf
# Yield: kg per nett hectare | Production: Metric Tonnes
# ============================================================================

YALA_2024_DATA = {
    'season': 'Yala',
    'year': 2024,
    'source': 'Yala2024Metric.pdf',
    'national_avg_yield_kg_ha': 3893,
    'national_total_production_mt': 1976496,
    'national_total_harvested_ha': 429744,
    'districts': {
        'COLOMBO':       {'yield_kg_ha': 1975, 'production_mt': 2582,   'harvested_ha': 1307},
        'GAMPAHA':       {'yield_kg_ha': 2831, 'production_mt': 13793,  'harvested_ha': 4872},
        'KALUTARA':      {'yield_kg_ha': 2694, 'production_mt': 14759,  'harvested_ha': 5478},
        'KANDY':         {'yield_kg_ha': 2859, 'production_mt': 19995,  'harvested_ha': 6990},
        'MATALE':        {'yield_kg_ha': 4500, 'production_mt': 43178,  'harvested_ha': 9595},
        'NUWARAELIYA':   {'yield_kg_ha': 3905, 'production_mt': 4690,   'harvested_ha': 1201},
        'GALLE':         {'yield_kg_ha': 2602, 'production_mt': 7658,   'harvested_ha': 2943},
        'MATARA':        {'yield_kg_ha': 2936, 'production_mt': 24687,  'harvested_ha': 8407},
        'HAMBANTOTA':    {'yield_kg_ha': 5381, 'production_mt': 165702, 'harvested_ha': 30794},
        'MANNAR':        {'yield_kg_ha': 2184, 'production_mt': 6210,   'harvested_ha': 2843},
        'VAVUNIYA':      {'yield_kg_ha': 4175, 'production_mt': 21731,  'harvested_ha': 5204},
        'MULATIVU':      {'yield_kg_ha': 4198, 'production_mt': 32502,  'harvested_ha': 7743},
        'KILLINOCHCHI':  {'yield_kg_ha': 4339, 'production_mt': 35463,  'harvested_ha': 8173},
        'BATTICALOA':    {'yield_kg_ha': 3589, 'production_mt': 106587, 'harvested_ha': 29701},
        'AMPARA':        {'yield_kg_ha': 5532, 'production_mt': 333935, 'harvested_ha': 60363},
        'TRINCOMALEE':   {'yield_kg_ha': 5053, 'production_mt': 135054, 'harvested_ha': 26730},
        'KURUNEGALA':    {'yield_kg_ha': 3708, 'production_mt': 210243, 'harvested_ha': 56696},
        'PUTTALAM':      {'yield_kg_ha': 3609, 'production_mt': 46600,  'harvested_ha': 12911},
        'ANURADHAPURA':  {'yield_kg_ha': 4863, 'production_mt': 241103, 'harvested_ha': 49580},
        'POLONNARUWA':   {'yield_kg_ha': 5716, 'production_mt': 336801, 'harvested_ha': 58927},
        'BADULLA':       {'yield_kg_ha': 4992, 'production_mt': 51667,  'harvested_ha': 10351},
        'MONARAGALA':    {'yield_kg_ha': 4334, 'production_mt': 69742,  'harvested_ha': 16091},
        'RATNAPURA':     {'yield_kg_ha': 4566, 'production_mt': 40096,  'harvested_ha': 8781},
        'KEGALLE':       {'yield_kg_ha': 2884, 'production_mt': 11718,  'harvested_ha': 4063},
    }
}

MAHA_2024_2025_DATA = {
    'season': 'Maha',
    'year': 2024,
    'source': '2024_2025Maha_Metric.pdf',
    'national_avg_yield_kg_ha': 3679,
    'national_total_production_mt': 2745298,
    'national_total_harvested_ha': 701453,
    'districts': {
        'COLOMBO':       {'yield_kg_ha': 2870, 'production_mt': 8051,   'harvested_ha': 2805},
        'GAMPAHA':       {'yield_kg_ha': 3302, 'production_mt': 27810,  'harvested_ha': 8423},
        'KALUTARA':      {'yield_kg_ha': 2727, 'production_mt': 24229,  'harvested_ha': 8884},
        'KANDY':         {'yield_kg_ha': 3480, 'production_mt': 30827,  'harvested_ha': 8858},
        'MATALE':        {'yield_kg_ha': 4507, 'production_mt': 72064,  'harvested_ha': 15989},
        'NUWARAELIYA':   {'yield_kg_ha': 3188, 'production_mt': 7826,   'harvested_ha': 2455},
        'GALLE':         {'yield_kg_ha': 2968, 'production_mt': 17198,  'harvested_ha': 5794},
        'MATARA':        {'yield_kg_ha': 3479, 'production_mt': 29671,  'harvested_ha': 8534},
        'HAMBANTOTA':    {'yield_kg_ha': 5725, 'production_mt': 180925, 'harvested_ha': 31606},
        'JAFFNA':        {'yield_kg_ha': 2246, 'production_mt': 19426,  'harvested_ha': 8647},
        'MANNAR':        {'yield_kg_ha': 4776, 'production_mt': 94978,  'harvested_ha': 19887},
        'VAVUNIYA':      {'yield_kg_ha': 3627, 'production_mt': 77250,  'harvested_ha': 21300},
        'MULATIVU':      {'yield_kg_ha': 2927, 'production_mt': 59691,  'harvested_ha': 20395},
        'KILLINOCHCHI':  {'yield_kg_ha': 2514, 'production_mt': 62033,  'harvested_ha': 24675},
        'BATTICALOA':    {'yield_kg_ha': 2526, 'production_mt': 156727, 'harvested_ha': 62040},
        'AMPARA':        {'yield_kg_ha': 3935, 'production_mt': 281814, 'harvested_ha': 71623},
        'TRINCOMALEE':   {'yield_kg_ha': 3905, 'production_mt': 175283, 'harvested_ha': 44888},
        'KURUNEGALA':    {'yield_kg_ha': 3982, 'production_mt': 323867, 'harvested_ha': 81343},
        'PUTTALAM':      {'yield_kg_ha': 3972, 'production_mt': 66930,  'harvested_ha': 16849},
        'ANURADHAPURA':  {'yield_kg_ha': 4263, 'production_mt': 435988, 'harvested_ha': 102284},
        'POLONNARUWA':   {'yield_kg_ha': 4794, 'production_mt': 295310, 'harvested_ha': 61598},
        'BADULLA':       {'yield_kg_ha': 3973, 'production_mt': 85428,  'harvested_ha': 21502},
        'MONARAGALA':    {'yield_kg_ha': 4139, 'production_mt': 145378, 'harvested_ha': 35121},
        'RATNAPURA':     {'yield_kg_ha': 4419, 'production_mt': 45085,  'harvested_ha': 10204},
        'KEGALLE':       {'yield_kg_ha': 3742, 'production_mt': 21509,  'harvested_ha': 5749},
    }
}

# ============================================================================
# WEATHER DATA FROM METEOROLOGICAL PDFs
# Source: 1.3.pdf (temperature), 1.5.pdf (humidity), 1.6.pdf (rainfall)
# Station-level annual averages (2018-2023)
# ============================================================================

STATION_TEMPERATURE = {
    # Source: 1.3.pdf - Annual average air temperature (°C) 2018-2023
    'Colombo':         {'avg_1961_1990': 27.4, 2018: 28.0, 2019: 28.2, 2020: 28.6, 2021: 28.1, 2022: 28.2, 2023: 28.0},
    'Jaffna':          {'avg_1961_1990': 27.9, 2018: 28.6, 2019: 28.6, 2020: 28.4, 2021: 28.3, 2022: 28.2, 2023: 28.5},
    'Trincomalee':     {'avg_1961_1990': 27.7, 2018: 28.2, 2019: 28.3, 2020: 28.0, 2021: 27.8, 2022: 28.1, 2023: 28.3},
    'Hambantota':      {'avg_1961_1990': 27.2, 2018: 28.0, 2019: 28.3, 2020: 28.5, 2021: 28.2, 2022: 28.3, 2023: 28.2},
    'Ratnapura':       {'avg_1961_1990': 27.5, 2018: 27.6, 2019: 27.9, 2020: 28.1, 2021: 27.5, 2022: 27.5, 2023: 27.8},
    'Anuradhapura':    {'avg_1961_1990': 27.6, 2018: 28.2, 2019: 28.6, 2020: 28.5, 2021: 28.4, 2022: 28.1, 2023: 28.3},
    'Katugastota':     {'avg_1961_1990': 25.4, 2018: 25.7, 2019: 25.8, 2020: 25.7, 2021: 25.5, 2022: 25.3, 2023: 25.8},
    'Bandarawela':     {'avg_1961_1990': 20.2, 2018: 20.8, 2019: 20.7, 2020: 20.2, 2021: 20.1, 2022: 20.5, 2023: 20.8},
    'NuwaraEliya':     {'avg_1961_1990': 15.8, 2018: 16.4, 2019: 16.5, 2020: 16.3, 2021: 16.0, 2022: 16.2, 2023: 16.3},
    'Kurunegala':      {'avg_1961_1990': 27.4, 2018: 28.0, 2019: 28.1, 2020: 27.8, 2021: 27.8, 2022: 27.6, 2023: 28.0},
    'Puttalam':        {'avg_1961_1990': 27.8, 2018: 28.8, 2019: 28.6, 2020: 28.2, 2021: 28.2, 2022: 28.6, 2023: 28.5},
    'Batticaloa':      {'avg_1961_1990': 27.7, 2018: 28.3, 2019: 28.3, 2020: 28.1, 2021: 28.0, 2022: 27.8, 2023: 28.4},
    'Katunayake':      {'avg_1961_1990': 27.2, 2018: 27.9, 2019: 28.2, 2020: 28.6, 2021: 28.1, 2022: 27.8, 2023: 28.1},
    'Ratmalana':       {'avg_1961_1990': 27.3, 2018: 27.8, 2019: 28.1, 2020: 28.5, 2021: 28.0, 2022: 27.9, 2023: 28.0},
    'Galle':           {'avg_1961_1990': 27.0, 2018: 27.8, 2019: 27.6, 2020: 27.8, 2021: 27.6, 2022: 27.6, 2023: 27.8},
    'Badulla':         {'avg_1961_1990': 23.4, 2018: 24.0, 2019: 24.0, 2020: 23.8, 2021: 23.5, 2022: 23.8, 2023: 24.0},
    'Mannar':          {'avg_1961_1990': 28.0, 2018: 28.7, 2019: 28.9, 2020: 28.5, 2021: 28.5, 2022: 28.4, 2023: 28.5},
    'Vavuniya':        {'avg_1961_1990': 27.8, 2018: 28.5, 2019: 28.6, 2020: 28.4, 2021: 28.3, 2022: 28.0, 2023: 28.3},
    'Polonnaruwa':     {'avg_1961_1990': 27.4, 2018: 28.1, 2019: 28.3, 2020: 28.0, 2021: 27.8, 2022: 27.8, 2023: 28.2},
    'Monaragala':      {'avg_1961_1990': 27.0, 2018: 27.6, 2019: 27.7, 2020: 27.2, 2021: 27.3, 2022: 27.5, 2023: 27.7},
    'Potuvil':         {'avg_1961_1990': 27.5, 2018: 28.0, 2019: 28.1, 2020: 27.8, 2021: 27.7, 2022: 27.8, 2023: 28.0},
    'Mahailluppallama':{'avg_1961_1990': 27.5, 2018: 28.3, 2019: 28.5, 2020: 28.3, 2021: 28.2, 2022: 28.0, 2023: 28.3},
    'Mattala':         {'avg_1961_1990': 27.1, 2018: 27.8, 2019: 28.2, 2020: 28.4, 2021: 28.1, 2022: 28.3, 2023: 28.1},
}

STATION_RAINFALL = {
    # Source: 1.6.pdf - Annual total rainfall (mm) 2018-2023
    'Colombo':         {'avg_1961_1990': 2423.8, 2018: 2562.2, 2019: 2864.8, 2020: 2083.7, 2021: 2856.1, 2022: 2390.1, 2023: 3393.0},
    'Jaffna':          {'avg_1961_1990': 1231.1, 2018: 1120.8, 2019: 1339.4, 2020: 1714.3, 2021: 2043.3, 2022: 1624.0, 2023: 1440.5},
    'Trincomalee':     {'avg_1961_1990': 1580.2, 2018: 2031.3, 2019: 1525.9, 2020: 1496.9, 2021: 1691.5, 2022: 1816.6, 2023: 2261.7},
    'Hambantota':      {'avg_1961_1990': 1049.6, 2018: 814.5,  2019: 1734.1, 2020: 717.5,  2021: 1207.1, 2022: 665.7,  2023: 2041.9},
    'Ratnapura':       {'avg_1961_1990': 3749.2, 2018: 3372.5, 2019: 3672.8, 2020: 3447.2, 2021: 4410.5, 2022: 4377.4, 2023: 4781.1},
    'Anuradhapura':    {'avg_1961_1990': 1284.6, 2018: 1450.7, 2019: 1324.2, 2020: 1262.2, 2021: 1533.4, 2022: 1558.4, 2023: 1652.9},
    'Katugastota':     {'avg_1961_1990': 1840.2, 2018: 2029.3, 2019: 1543.0, 2020: 1495.8, 2021: 2422.8, 2022: 1822.1, 2023: 2264.8},
    'Bandarawela':     {'avg_1961_1990': 1571.8, 2018: 1810.0, 2019: 1835.7, 2020: 1297.5, 2021: 1785.8, 2022: 1821.1, 2023: 2185.2},
    'NuwaraEliya':     {'avg_1961_1990': 1905.3, 2018: 2173.1, 2019: 1954.5, 2020: 1566.0, 2021: 1935.8, 2022: 1827.7, 2023: 2156.9},
    'Kurunegala':      {2019: 2317.4, 2020: 1769.4, 2021: 2012.8, 2022: 2539.0,  2023: 2020.0},
    'Puttalam':        {2019: 1419.2, 2020: 1733.8, 2021: 921.0,  2022: 1067.5,  2023: 970.1},
    'Batticaloa':      {2018: 1766.3, 2019: 2158.2, 2020: 1615.3, 2021: 1781.1, 2022: 1853.4, 2023: 1996.6},
    'Katunayake':      {2018: 2020.3, 2019: 2438.5, 2020: 2044.0, 2021: 2963.6, 2022: 1590.0, 2023: 3039.0},
    'Ratmalana':       {2018: 2488.2, 2019: 3066.5, 2020: 2042.4, 2021: 2520.2, 2022: 2526.5, 2023: 3303.3},
    'Galle':           {2018: 2283.9, 2019: 3024.3, 2020: 2198.0, 2021: 2634.7, 2022: 2713.8, 2023: 3942.2},
    'Badulla':         {2018: 1827.9, 2019: 1789.9, 2020: 1467.6, 2021: 1960.2, 2022: 1755.9, 2023: 2153.0},
    'Mannar':          {2018: 913.4,  2019: 1034.9, 2020: 921.3,  2021: 1720.3, 2022: 918.4,  2023: 1237.7},
    'Vavuniya':        {2018: 1550.1, 2019: 1427.7, 2020: 1324.5, 2021: 1527.9, 2022: 1545.7, 2023: 2046.7},
    'Polonnaruwa':     {2018: 1616.9, 2019: 1579.2, 2020: 1449.9, 2021: 1698.3, 2022: 1825.4, 2023: 2082.2},
    'Monaragala':      {2018: 1788.3, 2019: 2112.3, 2020: 1320.7, 2021: 2005.8, 2022: 2114.7, 2023: 2512.1},
    'Potuvil':         {2018: 1375.9, 2019: 1880.2, 2020: 1091.6, 2021: 1758.8, 2022: 1210.9, 2023: 1634.8},
    'Mahailluppallama':{2018: 1706.0, 2019: 1819.7, 2020: 1179.3, 2021: 1529.1, 2022: 1408.6, 2023: 1748.9},
    'Mattala':         {2018: 1384.6, 2019: 1318.8, 2020: 722.4,  2021: 1092.4, 2022: 1268.6, 2023: 2112.4},
    'Mullaitivu':      {2023: 2077.6},
}

STATION_HUMIDITY = {
    # Source: 1.5.pdf - Average relative humidity (%) Day values, 2019-2023
    'Colombo':         {2019: 73, 2020: 72, 2021: 74, 2022: 73, 2023: 74},
    'Trincomalee':     {2019: 65, 2020: 64, 2021: 66, 2022: 66, 2023: 66},
    'Hambantota':      {2019: 67, 2020: 68, 2021: 69, 2022: 69, 2023: 70},
    'Anuradhapura':    {2019: 62, 2020: 63, 2021: 65, 2022: 64, 2023: 65},
    'Batticaloa':      {2019: 75, 2020: 73, 2021: 74, 2022: 74, 2023: 76},
    'Katunayake':      {2019: 74, 2020: 73, 2021: 75, 2022: 74, 2023: 75},
    'Ratmalana':       {2019: 72, 2020: 71, 2021: 74, 2022: 73, 2023: 73},
    'Potuvil':         {2019: 74, 2020: 71, 2021: 73, 2022: 72, 2023: 73},
    'Polonnaruwa':     {2019: 65, 2020: 65, 2021: 67, 2022: 68, 2023: 68},
    'Monaragala':      {2019: 68, 2020: 67, 2021: 71, 2022: 70, 2023: 71},
    'Mattala':         {2019: 71, 2020: 68, 2021: 71, 2022: 70, 2023: 74},
    'Mullaitivu':      {2023: 69},
}

# Map districts to nearest weather stations
DISTRICT_TO_STATION = {
    'COLOMBO':      'Colombo',
    'GAMPAHA':      'Katunayake',
    'KALUTARA':     'Ratmalana',
    'KANDY':        'Katugastota',
    'MATALE':       'Katugastota',
    'NUWARAELIYA':  'NuwaraEliya',
    'GALLE':        'Galle',
    'MATARA':       'Hambantota',
    'HAMBANTOTA':   'Hambantota',
    'JAFFNA':       'Jaffna',
    'MANNAR':       'Mannar',
    'VAVUNIYA':     'Vavuniya',
    'MULATIVU':     'Trincomalee',
    'KILLINOCHCHI': 'Jaffna',
    'BATTICALOA':   'Batticaloa',
    'AMPARA':       'Potuvil',
    'TRINCOMALEE':  'Trincomalee',
    'KURUNEGALA':   'Kurunegala',
    'PUTTALAM':     'Puttalam',
    'ANURADHAPURA': 'Anuradhapura',
    'POLONNARUWA':  'Polonnaruwa',
    'BADULLA':      'Badulla',
    'MONARAGALA':   'Monaragala',
    'RATNAPURA':    'Ratnapura',
    'KEGALLE':      'Katugastota',
}

# ============================================================================
# DISEASE KNOWLEDGE FROM RRDI PDFs
# Source: Rice Research and Development Institute, Department of Agriculture
# 8 diseases: Bacterial Leaf Blight, Brown Spot, False Smut, Leaf Scald,
#   Narrow Brown Leaf Spot, Rice Blast, Sheath Blight, Sheath Rot
# ============================================================================

DISEASE_KNOWLEDGE = {
    'Bacterial Leaf Blight': {
        'source': 'RRDI_ricediseases_BacterialLeafBlight - Department of Agriculture Sri Lanka',
        'url': 'https://doa.gov.lk/rrdi_ricediseases_bacterialleafblight/',
        'causative_agent': 'Bacteria - Xanthomonas oryzae pv. oryzae',
        'affected_parts': 'Leaves',
        'affected_stages': 'Seedling stage and maturity stage',
        'symptoms': [
            'On seedling stage: wilting and yellowing of leaves, or wilting of seedlings (called kresek)',
            'Kresek on seedlings may sometimes be confused with early rice stem borer damage',
            'To distinguish kresek from stem borer damage: kresek should show yellowish bacterial ooze coming out of the cut ends when leaves are squeezed',
            'Unlike plants infested with stem borer, rice plants with kresek are not easily pulled out from soil',
            'On mature plants: lesions usually develop as water-soaked orange stripes on leaf blades or leaf tips',
            'Lesions have a wavy margin and progress toward the leaf base',
        ],
        'favorable_conditions': {
            'temperature': '25-34°C',
            'humidity': 'High humidity',
            'rainfall': 'Strong winds and continuous heavy rains',
            'other': 'Excessive application of nitrogen fertilizer; occurs in both irrigated and rainfed lowland areas',
        },
        'management': [
            'Application of urea in recommended dosages or application of urea based on leaf colour chart',
            'Ensure good drainage of fields',
            'Immediately after disease is observed, stop water supply and let the field dry',
            'When total removal of water is not possible, try to out-flow water through drainage ditches or water courses (wakkada)',
            'Water drained from infected fields should not be diverted through disease-free fields as much as possible',
            'Application of potassium fertilizer could manage further spread of the disease',
        ],
        'resistant_varieties': [],
        'susceptible_varieties': [],
        # 5 disease images from the RRDI PDF page (no alt text or captions on original page)
        'images': [
            {
                'filename': 'RRDI_blb.jpg',
                'url': 'https://doa.gov.lk/wp-content/uploads/2022/07/RRDI_blb.jpg',
                'description': 'Bacterial Leaf Blight leaf lesions',
                'size': '200x200',
            },
            {
                'filename': 'RRDI_blb1.jpg',
                'url': 'https://doa.gov.lk/wp-content/uploads/2022/07/RRDI_blb1.jpg',
                'description': 'Bacterial Leaf Blight leaf lesions or blight progression',
                'size': '300x300',
            },
            {
                'filename': 'RRDI_bbl4.jpg',
                'url': 'https://doa.gov.lk/wp-content/uploads/2022/07/RRDI_bbl4.jpg',
                'description': 'Bacterial Leaf Blight affected leaf',
                'size': '300x300',
            },
            {
                'filename': 'RRDI_kresek.jpg',
                'url': 'https://doa.gov.lk/wp-content/uploads/2022/07/RRDI_kresek.jpg',
                'description': 'Kresek (seedling wilt) symptoms of Bacterial Leaf Blight',
                'size': '300x300',
            },
            {
                'filename': 'RRDI_blb3.jpg',
                'url': 'https://doa.gov.lk/wp-content/uploads/2022/07/RRDI_blb3.jpg',
                'description': 'Bacterial Leaf Blight leaf lesions',
                'size': '300x300',
            },
            {
                'filename': 'RRDI_BacterialLeafBlight.png',
                'local_path': 'rice leaf diseases dataset/Bacterialblight/RRDI_BacterialLeafBlight.png',
                'description': 'RRDI PDF summary image (local)',
            },
        ],
    },

    'Brown Spot': {
                'images': [
                    {
                        'filename': 'RRDI_BrownSpot.png',
                        'local_path': 'rice leaf diseases dataset/Brownspot/RRDI_BrownSpot.png',
                        'description': 'RRDI PDF summary image (local)',
                    },
                ],
        'source': 'RRDI_ricediseases_BrownSpot - Department of Agriculture Sri Lanka',
        'url': 'https://doa.gov.lk/rrdi_ricediseases_brownspot/',
        'causative_agent': 'Fungus - Cochliobolus miyabeanus (Bipolaris oryzae)',
        'affected_parts': 'Coleoptiles, leaves, leaf sheath, immature florets, branches of the panicle, glumes, and grains',
        'affected_stages': 'Present at emergence; more prevalent as plants approach maturity and leaves senesce. Seed-borne disease.',
        'severity_note': 'Yield losses from leaf spots are probably not serious. Economic losses occur when the fungus attacks the panicle, including the grain.',
        'symptoms': [
            'Brown, circular to oval spots on coleoptiles which may lead to seedling blight',
            'Seedling blight may cause sparse or inadequate stands and feeble plants',
            'Spots are smaller on young leaves than on upper leaves',
            'Spots vary from minute dark (dark brown to reddish brown) to large oval/circular (dark brown margin with light reddish-brown or gray center)',
            'Spots on leaf sheath and hulls are similar to those on leaves',
            'Infected glumes show general black discoloration',
            'Infected immature florets: grain development hindered, or kernels are light weight or chalky',
        ],
        'favorable_conditions': {
            'temperature': '16-36°C',
            'humidity': '86-100% relative humidity',
            'rainfall': 'Drought conditions also favor disease',
            'other': 'Soils with low level of required nutrients or problem soils (high salinity, Iron toxicity)',
        },
        'management': [
            'Application of urea in recommended dosages or based on leaf colour chart',
            'Weed management',
        ],
        'next_season_management': [
            'Application of organic fertilizer to increase soil quality',
            'Use of certified seed paddy free from the disease',
            'Addition of burnt paddy husk (250 kg per acre) to soil during land preparation',
            'Abstain addition of disease-infected straw',
            'Treatment of seed paddy by dipping in hot water (53-54°C) for 10-12 minutes',
            'Treatment of seeds with a seed-protectant fungicide',
            'Crop rotation',
            'Proper land leveling',
        ],
        'resistant_varieties': [],
        'susceptible_varieties': ['Most varieties under nutrient stress conditions'],
    },

    'False Smut': {
                'images': [
                    {
                        'filename': 'False_Smut.png',
                        'local_path': 'rice leaf diseases dataset/reference_images/False_Smut.png',
                        'description': 'RRDI PDF summary image (local)',
                    },
                ],
        'source': 'RRDI_ricediseases_FlashSmut - Department of Agriculture Sri Lanka',
        'url': 'https://doa.gov.lk/rrdi_ricediseases_flashsmut/',
        'causative_agent': 'Fungus - Ustilaginoidea virens',
        'affected_parts': 'Mature panicle and grains',
        'affected_stages': 'Grain maturity stage; flowering stage',
        'severity_note': 'This disease is considered as an omen of a good harvest, as the factors that favour this disease will also favour a good yield.',
        'symptoms': [
            'Disease symptoms start when grains get mature; pericarp splits facilitating the causative agent',
            'Seed coat remains green while inside the grain, disease develops forming large orange to brown-green fruiting structures',
            'Could be observed in one or more grains of the mature panicle',
            'Later the orange covering ruptures exposing a mass of greenish-black spores',
            'The grain is then replaced by one or more sclerotia',
        ],
        'favorable_conditions': {
            'temperature': '25-35°C',
            'humidity': 'High humidity',
            'rainfall': 'Rain and winds during flowering stage',
            'other': 'Excessive use of nitrogen fertilizer; higher plant density; weeds',
        },
        'management': [
            'Application of urea in recommended dosages or based on leaf colour chart',
            'Water management',
        ],
        'next_season_management': [
            'Use of certified seed paddy free from the disease',
            'Weed management',
            'Addition of burnt paddy husk (253 kg per acre) to soil during land preparation',
            'Abstain addition of disease-infected straw',
            'Water management',
            'Maintaining an average level plant population in the field',
        ],
        'resistant_varieties': [],
        'susceptible_varieties': ['Bw 367', 'Bg 403'],
    },

    'Leaf Scald': {
                'images': [
                    {
                        'filename': 'Leaf_Scald.png',
                        'local_path': 'rice leaf diseases dataset/reference_images/Leaf_Scald.png',
                        'description': 'RRDI PDF summary image (local)',
                    },
                ],
        'source': 'RRDI_ricediseases_LeafScald - Department of Agriculture Sri Lanka',
        'url': 'https://doa.gov.lk/rrdi_ricediseases_leafscald/',
        'causative_agent': 'Fungus - Monographella albescens (Microdochium oryzae)',
        'affected_parts': 'Mature leaves, panicles, and seedlings',
        'affected_stages': 'Mature stage primarily. Seed-borne — survives between seasons/crops via infected seeds.',
        'severity_note': 'Common and sometimes severe in major rice growing districts in Sri Lanka.',
        'symptoms': [
            'Lesions start on leaf tips or from the edges of leaf blades',
            'Lesions have a chevron pattern of light (tan) and darker reddish-brown areas',
            'Leading edge of lesion is usually yellow to gold — causing fields to appear yellow or gold',
            'Lesions from edges of leaf blades have an indistinct, mottled pattern',
            'Affected leaves get dry and turn to straw-color',
            'Panicle infestation: uniform light to dark reddish-brown discoloration of entire florets or hulls of developing grain',
            'Can cause sterility or abortion of developing kernels',
        ],
        'favorable_conditions': {
            'temperature': 'Variable (common across Sri Lankan rice growing districts)',
            'humidity': 'Seed-borne pathogen — primary infection from infected seeds',
            'rainfall': 'Wet conditions favor spread',
            'other': 'Widespread across major rice growing districts in Sri Lanka',
        },
        'management': [
            'Application of urea in recommended dosages or based on leaf colour chart',
        ],
        'next_season_management': [
            'Use of certified seed paddy free from the disease',
            'Weed management',
            'Addition of burnt paddy husk (250 kg per acre) to soil during land preparation',
            'Abstain addition of disease-infected straw',
        ],
        'resistant_varieties': [],
        'susceptible_varieties': [],
    },

    'Narrow Brown Leaf Spot': {
                'images': [
                    {
                        'filename': 'Narrow_Brown_Leaf_Spot_1.png',
                        'local_path': 'rice leaf diseases dataset/reference_images/Narrow_Brown_Leaf_Spot_1.png',
                        'description': 'RRDI PDF summary image (local, 1)',
                    },
                    {
                        'filename': 'Narrow_Brown_Leaf_Spot_2.png',
                        'local_path': 'rice leaf diseases dataset/reference_images/Narrow_Brown_Leaf_Spot_2.png',
                        'description': 'RRDI PDF summary image (local, 2)',
                    },
                ],
        'source': 'RRDI_ricediseases_NarrowBrownLeafSpot - Department of Agriculture Sri Lanka',
        'url': 'https://doa.gov.lk/rrdi_ricediseases_narrowbrownleafspot/',
        'causative_agent': 'Fungus - Sphaerulina oryzina (Cercospora janseana)',
        'affected_parts': 'Leaf blades, leaf sheaths, pedicels, and glumes',
        'affected_stages': 'Severity varies year to year; more severe as plants approach maturity.',
        'severity_note': 'Severity varies based on level of varietal susceptibility. May cause severe leaf necrosis, premature ripening, yield reduction, and lodging.',
        'symptoms': [
            'Typical lesions: light to dark brown, linear, progressing parallel to the vein (2-10 mm long, 1-1.5 mm wide)',
            'On highly susceptible varieties: lesions may enlarge and connect together, forming brown linear necrotic regions',
            'On glumes: lesions are usually shorter but can be wider than those on leaves',
            'Brown lesions also found on pedicels',
            'Leaf sheath discoloration referred to as "net blotch" — netlike pattern of brown and light brown to yellow areas',
        ],
        'favorable_conditions': {
            'temperature': 'Variable severity year to year',
            'humidity': 'Increased severity as plants approach maturity',
            'rainfall': 'Wet conditions contribute',
            'other': 'Level of varietal susceptibility significantly affects severity',
        },
        'management': [
            'Application of urea in recommended dosages or based on leaf colour chart',
        ],
        'next_season_management': [
            'Application of organic fertilizer to improve soil',
            'Use of certified seed paddy free from the disease',
            'Weed management',
            'Addition of burnt paddy husk (250 kg per acre) to soil during land preparation',
            'Abstain addition of disease-infected straw',
        ],
        'resistant_varieties': [],
        'susceptible_varieties': [],
    },

    'Rice Blast': {
                'images': [
                    {
                        'filename': 'RRDI_RiceBlast.png',
                        'local_path': 'rice leaf diseases dataset/Rice Blast/RRDI_RiceBlast.png',
                        'description': 'RRDI PDF summary image (local)',
                    },
                ],
        'source': 'RRDI_ricediseases_RiceBlast - Department of Agriculture Sri Lanka',
        'url': 'https://doa.gov.lk/rrdi_ricediseases_riceblast/',
        'causative_agent': 'Fungus - Magnaporthe grisea (Pyricularia grisea)',
        'affected_parts': 'Leaves, nodes, panicles, seeds (named leaf blast, nodal blast, panicle blast, or neck blast based on part infected)',
        'affected_stages': 'Any life stage of the plant could be infected by this pathogen',
        'severity_note': 'Characteristics vary with crop life stage, susceptibility level of the cultivar, and environmental factors.',
        'symptoms': [
            'Spindle-shaped leaf spots with brown or reddish/yellowish-brown margins, ashy centers, and pointed ends',
            'Fully developed lesions: 1.0-1.5 cm in length and 0.3-0.5 cm in breadth',
            'Infected nodes become black and rot',
            'Infection of panicle base causes rotten neck/neck rot — panicle falls off',
            'Severe infection: secondary branches and grains affected resulting in partly filled grains, known as "whiteheads"',
        ],
        'favorable_conditions': {
            'temperature': 'Low temperature during night (17-20°C)',
            'humidity': 'High humidity',
            'rainfall': 'Foggy and dark climatic conditions',
            'other': 'Excessive application of nitrogen fertilizer; high densities of plants in the field',
        },
        'management': [
            'Application of urea in recommended dosages or based on leaf colour chart',
            'Weed management',
            'Tebuconazole 250g/l EC — dissolve 10 ml in 16 l of water (8-10 tanks per acre)',
            'Isoprothiolane 400g/l EC — dissolve 20 ml in 16 l of water (8-10 tanks per acre)',
            'Carbendazim 50% WP/WG — dissolve 11 g/11 ml in 16 l of water (8-10 tanks per acre)',
            'Tricyclazole 75% WP — dissolve 10 g in 16 l of water (8-10 tanks per acre)',
        ],
        'next_season_management': [
            'Use of resistant varieties: Bg 403, Bg 406, Bg 366, Bg 359, Bw 361, Bg 250',
            'Use of certified seed paddy free from the disease',
            'Addition of burnt paddy husk (250 kg per acre) to soil during land preparation',
            'Abstain addition of disease-infected straw',
        ],
        'resistant_varieties': ['Bg 403', 'Bg 406', 'Bg 366', 'Bg 359', 'Bw 361', 'Bg 250'],
        'susceptible_varieties': ['Bg 358', 'Bg 357', 'Bg 360', 'Bw 367', 'At 373', 'Bg 94/1'],
    },

    'Sheath Blight': {
                'images': [
                    {
                        'filename': 'Sheath_Blight.png',
                        'local_path': 'rice leaf diseases dataset/reference_images/Sheath_Blight.png',
                        'description': 'RRDI PDF summary image (local)',
                    },
                ],
        'source': 'RRDI_ricediseases_SheathBlight - Department of Agriculture Sri Lanka',
        'url': 'https://doa.gov.lk/rrdi_ricediseases_sheathblight/',
        'causative_agent': 'Fungi - Thanatephorus cucumeris (Rhizoctonia solani)',
        'affected_parts': 'Stem base (near water level), leaves, panicles',
        'affected_stages': 'Maximum tillering or flowering stage are more vulnerable',
        'severity_note': 'Infection occurs through infected plants or sclerotia which survive in soil for a long time depending on temperature and moisture levels. Infected straw, stubble, weeds could be other sources.',
        'symptoms': [
            'Spots/lesions initially develop near water level (flooded fields) or soil (upland fields) on the leaf sheath',
            'Lesions 1-3 cm long, oval or ellipsoidal initially, may enlarge and become irregular',
            'Mainly on leaf blade — white center, banded with green, brown, and orange coloration',
            'Advanced stages: flag leaf infection affects panicle exertion',
            'Sclerotia (asexual over-wintering structures) form on leaf sheath surface — 4-5 mm diameter, white when young, turn brown or purplish brown at maturity',
            'Sclerotia fall off onto soil surface and remain viable for years',
        ],
        'favorable_conditions': {
            'temperature': 'High temperature (up to 40°C)',
            'humidity': '>90% relative humidity',
            'rainfall': 'Foggy and dark climatic conditions',
            'other': 'Excessive use of nitrogen fertilizer; shade; high densities of plants in the field; varieties with higher number of tillers',
        },
        'management': [
            'Application of urea in recommended dosages or based on leaf colour chart',
            'Weed management',
            'Hexaconazole 50g/l EC — dissolve 32 ml in 16 l of water (8-10 tanks per acre)',
            'Propiconazole 250g/l EC — dissolve 16 ml in 16 l of water (8-10 tanks per acre)',
            'Thiophanate methyl 70% WP — dissolve 16 g in 16 l of water (8-10 tanks per acre)',
            'Pencicuron 25% WP — dissolve 32 g in 16 l of water (8-10 tanks per acre)',
            'Tebuconazole 250g/l EC — dissolve 10 ml in 16 l of water (8-10 tanks per acre)',
        ],
        'next_season_management': [
            'Use of certified seed paddy free from the disease',
            'Deep ploughing to bury infested plant residues into the soil',
            'Use of recommended seed rate: 2 bushels per acre (direct sowing)',
            'Maintaining an average level plant population in the field',
            'Addition of burnt paddy husk (250 kg per acre) to soil during land preparation',
            'Abstain addition of disease-infected straw',
        ],
        'resistant_varieties': [],
        'susceptible_varieties': ['Almost all rice varieties are susceptible to sheath blight in Sri Lanka. Varieties with clean plant base (less/no unproductive tillers) seem to escape the disease. No genetically resistant varieties exist.'],
    },

    'Sheath Rot': {
                'images': [
                    {
                        'filename': 'Sheath_Rot.png',
                        'local_path': 'rice leaf diseases dataset/reference_images/Sheath_Rot.png',
                        'description': 'RRDI PDF summary image (local)',
                    },
                ],
        'source': 'RRDI_ricediseases_SheathRot - Department of Agriculture Sri Lanka',
        'url': 'https://doa.gov.lk/rrdi_ricediseases_sheathrot/',
        'causative_agent': 'Fungus - Sarocladium oryzae',
        'affected_parts': 'Uppermost leaf sheaths that enclose the young panicle during the boot stage',
        'affected_stages': 'Boot stage (flowering to fruiting period)',
        'severity_note': 'Affects most prevailing rice varieties. Usually minor, affecting scattered tillers in a field. Occasionally, larger areas of a field may have significant damage.',
        'symptoms': [
            'Lesions are oblong or irregular oval spots with gray or light-brown centers and dark reddish-brown, diffuse margin (typical irregular target pattern)',
            'Usually expressed as reddish-brown discoloration of the flag-leaf sheath',
            'Early/severe infections affect panicle — causing partial emergence; un-emerged portion rots, turning florets red-brown to dark brown',
            'Grains from damaged panicles are discolored reddish-brown to dark brown and may not fill',
            'Powdery white growth (spores and hyphae) may be observed on the adaxial surface of affected sheaths',
        ],
        'favorable_conditions': {
            'temperature': '20-28°C (during flowering to fruiting period)',
            'humidity': 'High humidity',
            'rainfall': 'Wet conditions during boot stage',
            'other': 'Excessive use of nitrogen fertilizer',
        },
        'management': [
            'Application of urea in recommended dosages or based on leaf colour chart',
        ],
        'next_season_management': [
            'Application of organic fertilizer to improve soil conditions',
            'Use of certified seed paddy free from the disease',
            'Weed management',
            'Addition of burnt paddy husk (253 kg per acre) to soil during land preparation',
            'Abstain addition of straw infected with disease',
            'Management of insect populations (Sheath mite)',
        ],
        'resistant_varieties': [],
        'susceptible_varieties': ['Most prevailing rice varieties'],
    },
}


def get_district_yield(district, season='Yala'):
    """Get yield data for a district from PDF data"""
    district_upper = district.upper().replace(' ', '')
    if season == 'Yala':
        data = YALA_2024_DATA['districts'].get(district_upper)
    else:
        data = MAHA_2024_2025_DATA['districts'].get(district_upper)
    return data


def get_station_weather(station_name):
    """Get weather data for a station"""
    temp = STATION_TEMPERATURE.get(station_name, {})
    rain = STATION_RAINFALL.get(station_name, {})
    humid = STATION_HUMIDITY.get(station_name, {})
    return {'temperature': temp, 'rainfall': rain, 'humidity': humid}


def get_district_weather(district):
    """Get weather data for a district via nearest station"""
    district_upper = district.upper().replace(' ', '')
    station = DISTRICT_TO_STATION.get(district_upper)
    if station:
        return get_station_weather(station)
    return None


def get_all_districts():
    """Get list of all districts with data"""
    yala_districts = set(YALA_2024_DATA['districts'].keys())
    maha_districts = set(MAHA_2024_2025_DATA['districts'].keys())
    return sorted(yala_districts | maha_districts)


def get_disease_info(disease_name):
    """Get detailed disease information"""
    return DISEASE_KNOWLEDGE.get(disease_name)


def get_all_diseases():
    """Get list of all diseases"""
    return list(DISEASE_KNOWLEDGE.keys())
