package cic.cs.unb.ca.jnetpcap;

import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Date;

public class DateFormatter {
	
	public static String parseDateFromLong_old(long time, String format){
		try{
			if (format == null){
				format = "dd/MM/yyyy hh:mm:ss";					
			}
			SimpleDateFormat simpleFormatter = new SimpleDateFormat(format);
			Date tempDate = new Date(time);
			return simpleFormatter.format(tempDate);
		}catch(Exception ex){
			System.out.println(ex.toString());
			return "dd/MM/yyyy HH:mm:ss";
		}		
	}

	public static String parseDateFromLong(long time, String format){
		try{
			if (format == null){
				format = "dd/MM/yyyy HH:mm:ss.SSS";					
			}

			DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format);
        	LocalDateTime dateTime = LocalDateTime.ofInstant(
            	Instant.ofEpochMilli(time), 
            	ZoneId.systemDefault()
        	);

			return  dateTime.format(formatter);
		}catch(Exception ex){
			System.out.println(ex.toString());
			return "dd/MM/yyyy HH:mm:ss.SSS";
		}		
	}

	public static String convertMilliseconds2String_old(long time, String format) {

        if (format == null){
            format = "dd/MM/yyyy HH:mm:ss a";
        }

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format);
        LocalDateTime ldt = LocalDateTime.ofInstant(Instant.ofEpochMilli(time), ZoneId.systemDefault());
        return ldt.format(formatter);
	}

	public static String convertMilliseconds2String(long time, String format) {
        try{
			if (format == null){
				format = "dd/MM/yyyy HH:mm:ss.SSS";					
			}

			DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format);
        	LocalDateTime dateTime = LocalDateTime.ofInstant(
            	Instant.ofEpochMilli(time), 
            	ZoneId.systemDefault()
        	);

			return  dateTime.format(formatter);
		}catch(Exception ex){
			System.out.println(ex.toString());
			return "dd/MM/yyyy HH:mm:ss.SSS";
		}	
	}

	// public static long parseDateToLong(String dateStr, String format){
	// 	try{
	// 		if (format == null){
	// 			format = "dd/MM/yyyy hh:mm:ss";
	// 		}
	// 		SimpleDateFormat simpleFormatter = new SimpleDateFormat(format);
	// 		return simpleFormatter.parse(dateStr).getTime();
	// 	}
	// 	catch(Exception ex){
	// 		System.out.println(ex.toString());
	// 		return 0L;
	// 	}	
	// }

	// public static long parseDateToLong(String dateStr, String pattern) throws ParseException {
	// 	SimpleDateFormat sdf = new SimpleDateFormat(pattern);
	// 	Date date = sdf.parse(dateStr);
	// 	return date.getTime(); // trả về milliseconds từ epoch
	// }
}
