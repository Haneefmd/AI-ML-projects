import streamlit as st
from streamlit_option_menu import option_menu
import home, app3,app2
# Sidebar for navigation



class MultiApp:
    def __init__(self):
        self.apps = []
    
    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })
    def run():
        
        with st.sidebar:
            app = option_menu (
                menu_title = '📍Navigation',
                options = ['🏠Home','📈Sales Prediction Demo','📝User_Input Sales Prediction'],
                default_index=1,
                styles = {
                    "container":{"padding":"5!important", "background-color":"black"},
                    "icon":{"color":"white","font-size": "23px"},
                    "nav-link": {"color":"white","font-size":"20px","text-align":"left", "margin":"0px"},
                    "nav-link-selected":{"background-color":"#02ab21"},}

            )
            
            # Page routing based on selection
        if app == "🏠Home":
            home.app()
        elif app == "📈Sales Prediction Demo":
            app3.app()
        elif app == "📝User_Input Sales Prediction":
            app2.app()
    run()