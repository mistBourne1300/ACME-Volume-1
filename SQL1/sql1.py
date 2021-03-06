# sql1.py
"""Volume 1: SQL 1 (Introduction).
<Name>
<Class>
<Date>
"""

import matplotlib.pyplot as plt
import sqlite3 as sql
import csv
import numpy as np

# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn't exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of -1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    # open connection
    try:
        with sql.connect(db_file) as conn:
            # get cursor
            cur = conn.cursor()
            # execute commands
            cur.execute("DROP TABLE IF EXISTS MajorInfo;")
            cur.execute("DROP TABLE IF EXISTS CourseInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentGrades;")

            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT);")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT);")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER);")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGERS, Grade TEXT);")

            MajorInfo = [(1,'Math'), (2,'Science'), (3,'Writing'), (4, 'Art')]
            CourseInfo = [(1,'Calculus'), (2,'English'), (3,'Pottery'), (4,'History')]

            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", MajorInfo)
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", CourseInfo)

            with open("student_info.csv", 'r') as infile:
                rows = list(csv.reader(infile))
            cur.executemany("INSERT INTO StudentInfo VALUES (?,?,?);",rows)

            with open("student_grades.csv", 'r') as infile:
                rows = list(csv.reader(infile))

            cur.executemany("INSERT INTO StudentGrades VALUES (?,?,?);", rows)
            cur.execute("UPDATE StudentInfo SET MajorID=NULL WHERE MajorID==-1;")

    # close connection
    finally:
        conn.close()

def test_students():
    student_db()
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM StudentInfo;")
        print([d[0] for d in cur.description])
        for row in cur.execute("SELECT * FROM StudentInfo;"):
            print(row)
    



# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn't exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    # open connection
    try:
        with sql.connect(db_file) as conn:
            # get cursor
            cur = conn.cursor()
            # execute commands
            cur.execute("DROP TABLE IF EXISTS USEarthquakes;")
            cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL);")
            with open(data_file, 'r') as infile:
                rows = list(csv.reader(infile))
            cur.executemany("INSERT INTO USearthquakes VALUES (?,?,?,?,?,?,?,?,?)", rows)
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude==0;")
            cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day==0;")
            cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour==0;")
            cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute==0;")
            cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second==0;")

    # close connection
    finally:
        conn.close()

def test_earthquakes():
    earthquakes_db()
    with sql.connect("earthquakes.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM USEarthquakes"):
            print(row)
            if row[-1] == 0:
                raise RuntimeError(row)

# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    # list of the A students
    A_students = []
    # open connection
    try:
        with sql.connect(db_file) as conn:
            # get cursor
            cur = conn.cursor()
            # execute commands
            A_students += cur.execute(  "SELECT SI.StudentName, CI.CourseName "
                                        "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades AS SG "
                                        "WHERE SG.CourseID==CI.CourseID AND SI.StudentID==SG.StudentID AND SG.Grade=='A';")
            
            A_students += cur.execute(  "SELECT SI.StudentName, CI.CourseName "
                                        "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades AS SG "
                                        "WHERE SG.CourseID==CI.CourseID AND SI.StudentID==SG.StudentID AND SG.Grade=='A+';")

    # close connection
    finally:
        conn.close()
    return A_students

def test5():
    prob5()


# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    # two lists for the magnitudes of the respective centuries and one to average over
    magnitudes_19_century = []
    magnitudes_20_century = []
    all_magnitudes = []
    # open connection
    try: 
        with sql.connect(db_file) as conn:
            # get cursor
            cur = conn.cursor()
            # execute commands
            for mag in cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year>=1800 AND Year<1900;"):
                magnitudes_19_century.append(mag[0])
            for mag in cur.execute("SELECT Magnitude FROM USEarthquakes WHERE Year>=1900 AND Year<2000;"):
                magnitudes_20_century.append(mag[0])
            for mag in cur.execute("SELECT Magnitude FROM USEarthquakes;"):
                all_magnitudes.append(mag[0])
            
    # close connection
    finally:
        conn.close()

    # plot
    plt.subplot(121)
    plt.hist(magnitudes_19_century)
    plt.title("Earthquake Magnitudes in the 19th Century")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")

    plt.subplot(122)
    plt.hist(magnitudes_20_century)
    plt.title("Earthquake Magnitudes in the 20th Century")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")

    plt.show()
    return np.mean(all_magnitudes)



if __name__ == "__main__":
    # test_earthquakes()
    print(prob6())
    pass