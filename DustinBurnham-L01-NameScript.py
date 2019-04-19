# -*- coding: utf-8 -*-
"""
Dustin Burnham
Data Science 400
Assignment 1
Due: 1/8/19

This script when run will print the my name and return the current
date and time using the datetime.datetime package.
"""

import datetime as dt

def my_name():
    """
    my_name returns the string 'Dustin Burnham' (my name).
    """
    
    name = "Dustin Burnham"
    return(name)
    

def date_and_time():
    """
    date_and_time uses the datetime.datetime.now() package to get the current
    date and time and the datetime.strftime()function to put the date/time
    into the format month-day-year, hour:minute.  The date and time are 
    returned as a string.
    """
    
    date = dt.datetime.now()
    d_t = date.strftime("Date: %m-%d-%y, Time: %H:%M")
    return(d_t)
    
# Print the name
print(my_name())

# Print the date and time
print(date_and_time())