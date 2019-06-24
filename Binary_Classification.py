# -*- coding: utf-8 -*-


"""
Purpose: Binary clasification pipeline for use with ArcMap - [Python 2.7 32 bit]

Author: Michael Troyer
"""

from __future__ import division

try:
    import archook
    archook.get_arcpy()
    import arcpy
except:
    raise ImportError('Cannot find arcpy')    
    
import math
import datetime
import getpass
import os
import re
import sys
import traceback
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.svm             import LinearSVC
from sklearn.svm             import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from Exceptions import *
from Helpers import *
from Parameters import *


#--- Variables -------------------------------------------------------------------------------------------------------

start_time = datetime.datetime.now()
user = getpass.getuser()
err_banner = "#" * 40


#--- Settings --------------------------------------------------------------------------------------------------------

arcpy.env.addOutputsToMap = False
arcpy.env.overwriteOutput = True


######################################################################################################################
##
## MAIN PROGRAM
##
######################################################################################################################

def main(         
    input_analysis_area,                # Required | Polygon feature class path
    input_surveys,                      # Required | Polygon feature class path
    input_sites,                        # Required | Polygon feature class path
    input_analysis_features,            # Required | Dict: {feature path: feature field, ..}
    input_analysis_rasters,             # Required | List: [raster path,..]
    output_folder,                      # Required | Folder path
    input_surveys_select_field=None,    # Optional | Field name
    input_surveys_select_value=None,    # Optional | Field value
    input_sites_select_field=None,      # Optional | Field name
    input_sites_select_value=None,      # Optional | Field value
    n_non_site_sample_points=40000,     # Optional | integer
    n_site_sample_points=20000,         # Optional | integer
    min_site_sample_distance_m=15,      # Optional | integer (in meters)
    ):

    try:
        try:
            arcpy.CheckOutExtension('Spatial')
        except:
            raise LicenseError

#--- Create Geodatabase ----------------------------------------------------------------------------------------------

        # Clear memory JIC
        deleteInMemory()

        # Date/time stamp for outputs
        dt_stamp = re.sub('[^0-9]', '', str(datetime.datetime.now())[:16])

        # Output fGDB name and full path
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        gdb_name = "{}_{}".format(file_name, dt_stamp)
        gdb_path = os.path.join(output_folder, gdb_name + '.gdb')

        # Create a geodatabase
        arcpy.CreateFileGDB_management(output_folder, gdb_name, "10.0")

        # Set workspace to fGDB
        arcpy.env.workspace = gdb_name

        # Get input analysis area spatial reference for outputs
        spatial_ref = arcpy.Describe(input_analysis_area).spatialReference

        # Create feature datasets
        input_features_dataset = os.path.join(gdb_path, 'Input_Features')
        arcpy.CreateFeatureDataset_management(gdb_path, 'Input_Features', spatial_ref)

        print "Random Forest Modeling {}".format(datetime.datetime.now())
        print "Running environment: Python - {}".format(sys.version)
        print "User: {}".format(user)
        print "Output Location: {}".format(gdb_path)

#--- Get Anaysis Area ------------------------------------------------------------------------------------------------
                
        # Copy analysis area and get acres
        analysis_area = os.path.join(gdb_path, 'Analysis_Area')
        arcpy.CopyFeatures_management(input_analysis_area, analysis_area)
        analysis_acres_field, analysis_acres_total = get_acres(analysis_area)
        print '\nTotal acres within analysis area: {}\n'.format(round(analysis_acres_total, 4))

#--- Get Survey Data -------------------------------------------------------------------------------------------------

        # Get the input surveys FC as a feature layer clipped to analysis area
        arcpy.Clip_analysis(input_surveys, analysis_area, "in_memory\\in_surveys")

        # Get the surveys subset if given a subselection and copy to fGDB
        if input_surveys_select_field:
            for field in arcpy.Describe("in_memory\\in_surveys").fields:
                if field.name == input_surveys_select_field:                         
                    if field.type in ("Integer", "SmallInteger"):
                        where = '"{}" = {}'.format(input_surveys_select_field, input_surveys_select_value)
                    elif field.type == "String":
                        where = "{} = '{}'".format(input_surveys_select_field, input_surveys_select_value)
                    break
            print "Surveys subselection: WHERE [{}] = '{}'".format(case_field, case_value)
            arcpy.MakeFeatureLayer_management("in_memory\\in_surveys", "in_memory\\surveys_raw", where)
        # If no sub-selection, keep everything
        else:
            print ("No surveys subselection - using all records")
            arcpy.MakeFeatureLayer_management("in_memory\\in_surveys", "in_memory\\surveys_raw")

        # Dissolve and get survey acreage
        analysis_surveys = os.path.join(gdb_path, 'Analysis_Surveys')
        arcpy.Dissolve_management("in_memory\\surveys_raw", analysis_surveys)
        acres_field, survey_acres_total = get_acres(analysis_surveys)

        survey_coverage = survey_acres_total / analysis_acres_total          
        print ('Survey acres within analysis area: {}'.format(round(survey_acres_total, 2)))
        print ('Survey proportion within analysis area: {}\n'.format(round(survey_coverage, 3)))

        # Enforce minimum survey coverage for analysis
        if survey_coverage < 0.05:
            raise InsufficientSurveyCoverage

        arcpy.Delete_management("in_memory\\in_surveys")
        arcpy.Delete_management("in_memory\\in_surveys_raw")    

#--- Get Site Data ---------------------------------------------------------------------------------------------------

        analysis_sites =  os.path.join(gdb_path, 'Analysis_Sites')

        # Clip sites to survey coverage
        arcpy.Clip_analysis(input_sites, analysis_surveys, "in_memory\\in_sites")

        # Get the sites subset if given a subselection and copy to fGDB
        if input_sites_select_field:
            for field in arcpy.Describe("in_memory\\in_sites").fields:
                if field.name == input_surveys_select_field:                        
                    if field.type in ("Integer", "SmallInteger"):
                        where = '"{}" = {}'.format(input_surveys_select_field, input_sites_select_value)
                    elif field.type == "String":
                        where = "{} = '{}'".format(input_surveys_select_field, input_sites_select_value)
                    break
            print "Sites subselection: where [{}] = '{}'".format(input_surveys_select_field, input_sites_select_value)
            arcpy.MakeFeatureLayer_management("in_memory\\in_sites", "in_memory\\sites_raw", where)
        # If no sub-selection, keep everything
        else:
            print ("No sites subselection - using all records")
            arcpy.MakeFeatureLayer_management("in_memory\\in_sites", "in_memory\\sites_raw")
        
        arcpy.CopyFeatures_management("in_memory\\sites_raw", analysis_sites)

        site_count = int(arcpy.GetCount_management(analysis_sites).getOutput(0))
        site_density = round(site_count/survey_acres_total, 4)
        acres_per_site = round(1/site_density, 2)

        print 'Sites identified for analysis: {}'.format(site_count)
        print 'Site density in surveyed areas (sites/acre): {}'.format(site_density)
        print 'Approximately 1 site every {} acres\n'.format(acres_per_site)

        if site_count < 30:
            raise InsufficientSiteSample

        arcpy.Delete_management("in_memory\\sites")
        arcpy.Delete_management("in_memory\\sites_raw")

#--- Create Sample Dataset -------------------------------------------------------------------------------------------

        non_site_points = os.path.join(gdb_path, 'Sample_Points_Non_Site')
        site_points = os.path.join(gdb_path, 'Sample_Points_Site')
        raw_point_dataset = os.path.join(gdb_path, 'Raw_Points_Dataset')
        final_point_dataset = os.path.join(gdb_path, 'Analysis_Points_Dataset')

        print ('Creating point cloud..')

        desc = arcpy.Describe(analysis_area)
        xmin = desc.extent.XMin
        xmax = desc.extent.XMax
        ymin = desc.extent.YMin
        ymax = desc.extent.YMax

        # Create fishnet
        spacing = int(np.sqrt(n_non_site_sample_points))
        arcpy.CreateFishnet_management(
            out_feature_class=non_site_points,
            origin_coord="{} {}".format(xmin, ymin),  # lower left
            y_axis_coord="{} {}".format(xmin, ymin+10),  # lower left up ten meters
            number_rows=spacing,
            number_columns=spacing,
            corner_coord="{} {}".format(xmax, ymax),  # upper right
            labels="LABELS",
            template=analysis_area,
            geometry_type="POLYLINE"
            )
        # Dump the polyline feature and point non_site_points to the pt fc
        arcpy.Delete_management(non_site_points)  # This is the polyline
        non_site_points += '_label'  # Now its the points

        arcpy.CreateRandomPoints_management(
            out_path=gdb_path,
            out_name=os.path.basename(site_points),
            constraining_feature_class=analysis_sites,
            number_of_points_or_field=n_site_sample_points,
            minimum_allowed_distance="{} Meters".format(min_site_sample_distance_m),
            )

        # Create the template for the final points fc
        arcpy.CreateFeatureclass_management(
            out_path=gdb_path,
            out_name=os.path.basename(raw_point_dataset),
            geometry_type='POINT',
            spatial_reference=non_site_points
            )

        # Add class ids to site and not site points
        for fc in [site_points, non_site_points, raw_point_dataset]:
            arcpy.AddField_management(fc, 'Class', "SHORT")

        # TODO: don't encode unsurveyed areas
        arcpy.CalculateField_management(non_site_points, "Class", 0, "PYTHON_9.3")
        arcpy.CalculateField_management(site_points, "Class", 1, "PYTHON_9.3")

        # select by location - analysis points within a site need to be coded 1
        non_site_points_lyr = arcpy.MakeFeatureLayer_management(non_site_points, 'in_memory\\non_site_points')
        arcpy.SelectLayerByLocation_management(non_site_points_lyr, "INTERSECT", analysis_sites)
        arcpy.CalculateField_management(non_site_points_lyr, "Class", 1, "PYTHON_9.3")  # These ones are sites

        # Append site_points and non_site_points to final_poit_dataset and delete
        arcpy.Append_management([site_points, non_site_points], raw_point_dataset, "NO_TEST")

        # Encode if final points are within a survey or not
        arcpy.AddField_management(raw_point_dataset, 'Surveyed', "SHORT")
        arcpy.CalculateField_management(raw_point_dataset, "Surveyed", 0, "PYTHON_9.3")
        raw_points_lyr = arcpy.MakeFeatureLayer_management(raw_point_dataset, 'in_memory\\raw_point_dataset')
        arcpy.SelectLayerByLocation_management(raw_points_lyr, "INTERSECT", analysis_surveys)
        arcpy.CalculateField_management(raw_points_lyr, "Surveyed", 1, "PYTHON_9.3")

        # Delete the polyline feature and rename point feature class
        arcpy.Delete_management(non_site_points)
        arcpy.Delete_management(site_points)
        
        arcpy.Delete_management('in_memory\\non_site_points')
        arcpy.Delete_management('in_memory\\raw_point_dataset')

#--- Data Attribution Loops ------------------------------------------------------------------------------------------

        # Feature Classes
        feature_class_inputs = list(input_analysis_features.keys())
        analysis_fields = list(input_analysis_features.values()) + ['Class', 'Surveyed']

        for input_fc_path, field in input_analysis_features.items():
            fc_name = os.path.basename(str(input_fc_path))

            print 'Getting data from feature class: {}.{}'.format(fc_name, field)

            out_path = os.path.join(input_features_dataset, fc_name)
            arcpy.Clip_analysis(input_fc_path, analysis_area, out_path)

        feature_class_point_data = os.path.join(input_features_dataset, 'feature_class_data')
        arcpy.Intersect_analysis(
            in_features=feature_class_inputs + [raw_point_dataset],
            out_feature_class=final_point_dataset,
            )

        # clean up fields
        drop_fields = [f for f in arcpy.ListFields(final_point_dataset, "*") if f.name not in analysis_fields]
        for drop_field in drop_fields:
            try:
                arcpy.DeleteField_management(final_point_dataset, drop_field.name)
            except:
                pass
                # print ('Did not delete field: [{}]'.format(drop_field.name))

        arcpy.Delete_management(raw_point_dataset)

        # Rasters
        raster_clips = []
                
        for raster in input_analysis_rasters:
            raster_name = os.path.basename(raster)
            out_raster = os.path.join(gdb_path, 'raster_' + raster_name)

            print 'Getting data from raster: {}'.format(raster_name)

            arcpy.Clip_management(
                in_raster=raster,
                out_raster=out_raster,
                in_template_dataset=analysis_area,
                clipping_geometry="ClippingGeometry",
                )
            raster_clips.append([raster, raster_name])

            analysis_fields.append(raster_name)

        arcpy.sa.ExtractMultiValuesToPoints(final_point_dataset, raster_clips)

#--- Prepare Models --------------------------------------------------------------------------------------------------

        df = feature_class_to_pandas_data_frame(final_point_dataset, analysis_fields, null_value=-99999)

        # One hot encode categorical variables
        df, encode_columns = auto_one_hot(df)
        df = df.drop(encode_columns, axis=1)

        # Isolate the unsurveyed areas as extrapolation set
        extrapolation_set = df[df.Surveyed == 0]
        extrapolation_set = extrapolation_set.drop(['Class', 'Surveyed'], axis=1)

        # Split the Xs and Ys into train and test sets
        modeling_set = df[df.Surveyed == 1]
        modeling_set_Ys = modeling_set['Class']
        modeling_set_Xs = modeling_set.drop(['Class', 'Surveyed'], axis=1)
        trnX, tstX, trnY, tstY = train_test_split(modeling_set_Xs, modeling_set_Ys, test_size=0.3, random_state=42)

        # Prep classifiers
        classifiers = {
            'K Nearest Neighbors'                   : KNeighborsClassifier(),
            'Random Forest'                         : RandomForestClassifier(),
            'Gradient Boosted Decision Trees'       : GradientBoostingClassifier(),
            'Logistic Regression'                   : LogisticRegression(),    
            'Neural Network Multi-layer Perceptron' : MLPClassifier(),
            'Linear Support Vector Classifier'      : LinearSVC(),
            'Support Vector Machine'                : SVC(),
            }

#--- Evaluate Model --------------------------------------------------------------------------------------------------

        print '\n', ' Model Results '.center(60, '-'), '\n'

        cv = 10
        performance = {}

        for name, model in classifiers.items():

            scores = cross_val_score(model, trnX, trnY, cv=cv)
            mean = scores.mean(); stdev = scores.std()
            print "\n{}:\n\n\tAccuracy: {:.12f} (+/- {:.3f})".format(name, mean, stdev * 2)

            performance[name] = (mean, stdev)

            print '\n', '\t\t', 'Params'.center(45, '-')
            for k, v in sorted(model.get_params().items()):
                print '\t\t|', k.ljust(25, '.'), str(v)[:15].rjust(15, '.'), '|'
            print '\t\t', ''.center(45, '-'), '\n'

        ranked_models = sorted(performance.items(), key=lambda x: x[1][0], reverse=True)

#--- Tune Best Model -----------------------------------------------------------------------------------------------

        best_classifier_name, (best_mean, best_std) = ranked_models[0]
        best_classifier = classifiers[best_classifier_name]
        
        print ' {} Gridsearch '.format(best_classifier_name).center(60, '#')    
        print "Baseline Accuracy: {:.12f} (+/- {:.3f})".format(best_mean, best_std * 2)  

        gscv = GridSearchCV(best_classifier, parameters[best_classifier_name], n_jobs=-1, cv=cv, error_score=0)
        gscv.fit(trnX, trnY)

        # TODO: Write to csv
        print pd.DataFrame(gscv.cv_results_)

#--- Validate Tuned Model --------------------------------------------------------------------------------------------

        hypertuned_model = eval(str(gscv.best_estimator_))
        # Train on the whole data set less final validation set
        hypertuned_model.fit(trnX, trnY)
        predicted_y = hypertuned_model.predict(tstX)
        
        print
        print " {} Out-of-Sample Testing ".format(best_classifier_name).center(60, '#')
        print
        print "Accuracy Score: {}".format(accuracy_score(tstY, predicted_y))
        print
        print "Confusion Matrix:"
        print "|TN|FP|"
        print "|FN|TP|"
        print confusion_matrix(tstY, predicted_y)
        print
        print "Class Report:"
        print classification_report(tstY, predicted_y)

#--- Model Unsurveyed Areas ------------------------------------------------------------------------------------------

#--- Output Results to ArcMap ----------------------------------------------------------------------------------------

#--- House Keeping ---------------------------------------------------------------------------------------------------

        arcpy.CheckInExtension('Spatial')

    except arcpy.ExecuteError:
        print arcpy.GetMessages(2)
    except InsufficientSurveyCoverage:
        pass
    except InsufficientSiteSample:
        pass    
    except LicenseError:
        pass              
    except:
        pass

        
#####-----------------------------------------------------------------------------------------------------------------
##
## MAIN
##
#####-----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    gdb = r'C:\Users\mtroyer\Downloads\Local_Model_Testing\Modeling_Data.gdb'

    input_analysis_area = os.path.join(gdb, 'Analysis_Area')
    input_surveys = os.path.join(gdb, 'Analysis_Surveys')
    input_sites = os.path.join(gdb, 'Analysis_Sites')

    input_analysis_features = {
        os.path.join(gdb, 'Input_Features', 'Geology'): 'GEOLOGY_label',
        os.path.join(gdb, 'Input_Features', 'Soils'): 'SOILS_musym',
        os.path.join(gdb, 'Input_Features', 'Vegetation'): 'VEGETATION_type',
    }

    input_analysis_rasters = [
        os.path.join(gdb, 'raster_aspect_100m'),
        os.path.join(gdb, 'raster_slope_100m'),
    ]

    output_folder = r'C:\Users\mtroyer\Downloads\Local_Model_Testing\Outputs'

    # input_surveys_select_field = r''
    # input_surveys_select_value = r''

    # input_sites_select_field = r''
    # input_sites_select_value = r''

    main(
        input_analysis_area=input_analysis_area,
        input_surveys=input_surveys,
        input_sites=input_sites,
        input_analysis_features=input_analysis_features,
        input_analysis_rasters=input_analysis_rasters,
        output_folder=output_folder,
        input_surveys_select_field=None,
        input_surveys_select_value=None,
        input_sites_select_field=None,
        input_sites_select_value=None,
    )