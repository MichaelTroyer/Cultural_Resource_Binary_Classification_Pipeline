import pandas as pd
try:
    import archook
    archook.get_arcpy()
    import arcpy
except:
    raise ImportError('Cannot find arcpy') 


def deleteInMemory():
    """
    Delete in memory tables and feature classes
    reset to original worksapce when done
    """
    # get the original workspace
    orig_workspace = arcpy.env.workspace

    # Set the workspace to in_memory
    arcpy.env.workspace = "in_memory"
    # Delete all in memory feature classes
    for fc in arcpy.ListFeatureClasses():
        try:
            arcpy.Delete_management(fc)
        except: pass
    # Delete all in memory tables
    for tbl in arcpy.ListTables():
        try:
            arcpy.Delete_management(tbl)
        except: pass
    # Reset the workspace
    arcpy.env.workspace = orig_workspace


def buildWhereClauseFromList(table, field, valueList):
    """
    Takes a list of values and constructs a SQL WHERE
    clause to select those values within a given field and table.
    """
    # Add DBMS-specific field delimiters
    fieldDelimited = arcpy.AddFieldDelimiters(arcpy.Describe(table).path, field)
    
    # Determine field type
    fieldType = arcpy.ListFields(table, field)[0].type
    
    # Add single-quotes for string field values
    if str(fieldType) == 'String':
        valueList = ["'%s'" % value for value in valueList]
        
    # Format WHERE clause in the form of an IN statement
    whereClause = "%s IN(%s)" % (fieldDelimited, ', '.join(map(str, valueList)))
    return whereClause


def get_acres(fc):
    """
    Check for an acres field in fc - create if doesn't exist or flag for calculation.
    Recalculate acres and return name of acre field
    """    
    # Add ACRES field to analysis area - check if exists
    field_list = [field.name for field in arcpy.ListFields(fc) if field.name.upper() == "ACRES"]
    
    # If ACRES/Acres/acres exists in table, flag for calculation instead
    if field_list:
        acre_field = field_list[0] # select the 'acres' variant
    else:
        arcpy.AddField_management(fc, "ACRES", "DOUBLE", 15, 2)
        acre_field = "ACRES"
    arcpy.CalculateField_management(fc, acre_field, "!shape.area@ACRES!", "PYTHON_9.3")
    acres_sum = sum([row[0] for row in arcpy.da.SearchCursor(fc, acre_field)])
    return acre_field, acres_sum


def feature_class_to_pandas_data_frame(feature_class, field_list, null_value=-99999):
    """
    Load data into a Pandas Data Frame for subsequent analysis.
    :param feature_class: Input ArcGIS Feature Class.
    :param field_list: Fields for input.
    :return: Pandas DataFrame object.
    https://joelmccune.com/arcgis-to-pandas-data-frame/
    """
    return pd.DataFrame(
        arcpy.da.FeatureClassToNumPyArray(
            in_table=feature_class,
            field_names=field_list,
            skip_nulls=False,
            null_value=null_value
        )
    )


def auto_one_hot(df):
    encode_columns = list(df.select_dtypes(include=['category','object']))
    for column in encode_columns:
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df, encode_columns