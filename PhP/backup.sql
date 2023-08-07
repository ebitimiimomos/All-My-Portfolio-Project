

DROP TABLE IF EXISTS `training_sessions`;

CREATE TABLE `training_sessions` (
  `id` int(11) NOT NULL,
  `topic` varchar(255) DEFAULT NULL,
  `day` varchar(255) DEFAULT NULL,
  `time` varchar(255) DEFAULT NULL,
  `capacity` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


--
-- Dumping data for table `training_sessions`
--

LOCK TABLES `training_sessions` WRITE;

INSERT INTO `training_sessions` VALUES (1,'Email','Tuesday','12:00',3),(2,'Email','Wednesday','10:00',3),(3,'Email','Thursday','11:00',3),(4,'Library Use','Wednesday','11:00',2),(5,'Presentation Software','Tuesday','10:00',2),(6,'Presentation Software','Thursday','12:00',2),(7,'Spreadsheets','Tuesday','11:00',3),(8,'Spreadsheets','Wednesday','12:00',3),(9,'Spreadsheets','Thursday','10:00',3),(10,'Word Processing','Tuesday','10:00',4),(11,'Word Processing','Wednesday','11:00',4),(12,'Word Processing','Thursday','12:00',4);

UNLOCK TABLES;

--
-- Table structure for table `booking_request`
--

DROP TABLE IF EXISTS `booking_request`;

CREATE TABLE `booking_request` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `topic` varchar(255) DEFAULT NULL,
  `time` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `booking_request`
--

LOCK TABLES `booking_request` WRITE;

UNLOCK TABLES;

