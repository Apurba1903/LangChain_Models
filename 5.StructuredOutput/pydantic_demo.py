from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    
    name: str = 'Apurba'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=4, default= 2, description='A decimal value representing a CGPA of a Student')

new_student = {
    'name': 'Apurba',
    'age': 34,
    'email': 'abc@gmail.com',
    'cgpa': 3.50
}

student = Student(**new_student)

student_dict = dict(student)
print(student_dict)


student_json = student.model_dump_json()
print(student_json)