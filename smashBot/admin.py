from django.contrib import admin
from .models import AgentHyperparameters, Memory
# Register your models here.

admin.site.register(AgentHyperparameters)
admin.site.register(Memory)
