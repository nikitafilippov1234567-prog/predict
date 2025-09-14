from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse, Http404, FileResponse
from django.conf import settings
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib import messages
from django.urls import reverse
from .models import UploadedFile
import os
import uuid
import logging

logger = logging.getLogger(__name__)

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def login_view(request):
    if request.method == 'POST':
        password = request.POST.get('password')
        if password == settings.SITE_PASSWORD:
            request.session['authenticated'] = True
            logger.info(f"Successful authentication from IP: {get_client_ip(request)}")
            return redirect('upload')
        else:
            logger.warning(f"Failed authentication attempt from IP: {get_client_ip(request)}")
            return render(request, 'login.html', {'error': 'Неверный пароль'})
    
    return render(request, 'login.html')

def upload_view(request):
    if not request.session.get('authenticated'):
        return redirect('login')
    
    uploaded_files = UploadedFile.objects.all()[:10]  # Показываем последние 10 файлов
    
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            uploaded_file = request.FILES['file']
            
            # Генерируем уникальное имя файла
            file_extension = os.path.splitext(uploaded_file.name)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            relative_file_path = f"uploads/{unique_filename}"  # Относительный путь: uploads/abc123.txt

            # Сохраняем файл
            with default_storage.open(relative_file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Сохраняем информацию в базу данных
            file_record = UploadedFile.objects.create(
                original_name=uploaded_file.name,
                file_path=relative_file_path,  # Сохраняем относительный путь
                file_size=uploaded_file.size,
                uploader_ip=get_client_ip(request)
            )
            
            logger.info(f"File uploaded successfully: {uploaded_file.name} by IP: {get_client_ip(request)}")
            messages.success(request, f'Файл "{uploaded_file.name}" успешно загружен!')
            
            return redirect('upload')
            
        except PermissionError as e:
            logger.error(f"Permission error during upload: {str(e)}")
            messages.error(request, 'Ошибка: Нет прав для сохранения файла.')
        except OSError as e:
            logger.error(f"OS error during upload: {str(e)}")
            messages.error(request, 'Ошибка: Проблема с файловой системой.')
        except Exception as e:
            logger.error(f"Unexpected upload error: {str(e)}")
            messages.error(request, 'Ошибка при загрузке файла. Попробуйте снова.')
    
    return render(request, 'upload.html', {
        'uploaded_files': uploaded_files
    })

def download_file(request, file_id):
    file_record = get_object_or_404(UploadedFile, id=file_id)
    
    if not file_record.file_exists():
        raise Http404("Файл не найден")
    
    # Увеличиваем счетчик скачиваний
    file_record.download_count += 1
    file_record.save()
    
    logger.info(f"File downloaded: {file_record.original_name} by IP: {get_client_ip(request)}")

    full_file_path = os.path.join(settings.MEDIA_ROOT, file_record.file_path)
    response = FileResponse(
        open(full_file_path, 'rb'),
        as_attachment=True,
        filename=file_record.original_name
    )
    return response

def delete_file(request, file_id):
    if not request.session.get('authenticated'):
        return redirect('login')
    
    file_record = get_object_or_404(UploadedFile, id=file_id)
    original_name = file_record.original_name
    
    try:
        file_record.delete_file()
        messages.success(request, f'Файл "{original_name}" удален!')
        logger.info(f"File deleted: {original_name} by IP: {get_client_ip(request)}")
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        messages.error(request, 'Ошибка при удалении файла.')
    
    return redirect('upload')
