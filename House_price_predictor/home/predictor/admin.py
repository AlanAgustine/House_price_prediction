from django.contrib import admin
from .models import *


# Register your models here.
@admin.register(CarouselImages)
class CarouselImagesAdmin(admin.ModelAdmin):
    list_display = ("photo", "description")
    fields = ["photo", "description"]
