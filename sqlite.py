import sqlite3

## Inititalize Connection
conn = sqlite3.connect('test.db')

## Initiate Cursor
c = conn.cursor()


## Create the table
table_info = """
CREATE table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25),
SECTION VARCHAR(25), MARKS INT)
"""

c.execute(table_info)

## Add some data to the table
c.execute('''Insert Into STUDENT values ('Aditya', 'AIML', 'C',27)''')
c.execute('''Insert Into STUDENT values ('Vaidehi', 'AIML', 'C',21)''')
c.execute('''Insert Into STUDENT values ('Atharva', 'IT', 'A',26)''')
c.execute('''Insert Into STUDENT values ('Pooja', 'DS', 'A',16)''')
c.execute('''Insert Into STUDENT values ('Kamlesh', 'AIML', 'B',43)''')

## Display the data
print("The inserted records are")
data = c.execute('''Select * from STUDENT''')

for row in data:
    print(row)

## Commit the changes made
conn.commit()
conn.close()
