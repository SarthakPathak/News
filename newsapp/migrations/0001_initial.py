# Generated by Django 4.0.3 on 2022-04-24 12:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='historical_database',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source', models.CharField(max_length=50, null=True)),
                ('author', models.CharField(max_length=50, null=True)),
                ('title', models.CharField(max_length=500, null=True)),
                ('publishedAt', models.CharField(max_length=40, null=True)),
                ('content', models.CharField(max_length=500, null=True)),
                ('description', models.CharField(max_length=500, null=True)),
            ],
        ),
    ]
