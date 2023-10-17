from django.db import models


# Create your models here.
class CarouselImages(models.Model):
    photo = models.ImageField(upload_to="photo", blank=True, null=True)
    description = models.CharField(max_length=300)

    def __str__(self):
        return self.description
