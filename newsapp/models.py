from django.db import models

# Create your models here.
class historical_database(models.Model):
    source=models.CharField(max_length=50,null=True)
    author=models.CharField(max_length=50,null=True)
    title=models.CharField(max_length=500,null=True)
    publishedAt=models.CharField(max_length=40,null=True)
    content=models.CharField(max_length=500,null=True)
    description=models.CharField(max_length=500,null=True)

    def __str__(self):
        return str(self.id) + ',' + self.author + ',' + self.title + ',' + self.publishedAt + ',' + self.content + ',' + self.description
