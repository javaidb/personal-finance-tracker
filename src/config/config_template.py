'''
To assign what is considered RENT, processing functions will look for rent ranges at earlier and later regions per month.
In order to do this, you need to initialize a min/max of different rent costs that we can look for.
** Different rent ranges are really just in case of moving and a MIN/MAX in case of rent increases.
Format is as follows:
[
    {
        "min": <ENTER MIN RENT FOR PROPERTY>,
        "max": <ENTER MAX RENT FOR PROPERTY>)
    },
    {...PROPERTY 2...},
    ...
]
'''

rent_ranges = [
    {
        "min": "ENTER MINIMUM RENT PER PROPERTY",
        "max": "ENTER MAXIMUM RENT PER PROPERTY (UTILITIES, BASE RENT+ RENT INCREASE, ETC.)"
    }
]