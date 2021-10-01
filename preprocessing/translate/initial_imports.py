from argostranslate import package, translate
import datetime, csv

# Setting up translator
package.install_from_path('models/en_de.argosmodel')
installed_languages = translate.get_installed_languages()
translation_en_de = installed_languages[0].get_translation(installed_languages[1])

path2datasets = '../../data/datasets/'
